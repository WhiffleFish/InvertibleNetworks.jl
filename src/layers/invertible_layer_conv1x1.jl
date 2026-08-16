# 1x1 convolution operator using Householder matrices.
# Adapted from Putzky and Welling (2019): https://arxiv.org/abs/1911.10914
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020
#

export Conv1x1

"""
    C = Conv1x1(k; logdet=false)

 or

    C = Conv1x1(v1, v2, v3; logdet=false)

 Create network layer for 1x1 convolutions using Householder reflections.

 *Input*:

 - `k`: number of channels

 - `v1`, `v2`, `v3`: Vectors from which to construct matrix.

 - `logdet`: if true, returns logdet in forward pass (which is always zero)

 *Output*:

 - `C`: Network layer for 1x1 convolutions with Householder reflections.

 *Usage:*

 - Forward mode: `Y, logdet = C.forward(X)`

 - Inverse mode: `X = C.inverse(Y)`

 - Backward mode: `ΔX, X = C.backward(ΔY, Y)`

 *Trainable parameters:*

 - Householder vectors `C.v1`, `C.v2`, `C.v3`

 See also: [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct Conv1x1{P<:Parameter} <: NeuralNetLayer
    k::Int64
    v1::P
    v2::P
    v3::P
    logdet::Bool
    freeze::Bool
end

Flux.@layer Conv1x1

supports_per_sample_logdet(::Conv1x1) = true

# Constructor with random initializations
function Conv1x1(k;freeze=false, logdet=false)
    v1 = Parameter(glorot_uniform(k))
    v2 = Parameter(glorot_uniform(k))
    v3 = Parameter(glorot_uniform(k))
    return Conv1x1(k, v1, v2, v3, logdet, freeze)
end

function Conv1x1(v1, v2, v3;freeze=false, logdet=false)
    k = length(v1)
    v1 = Parameter(v1)
    v2 = Parameter(v2)
    v3 = Parameter(v3)
    return Conv1x1(k, v1, v2, v3, logdet, freeze)
end

function partial_derivative_outer(v::AbstractArray{T, 1}) where T
    k = length(v)
    out1 = v * v'
    n = v' * v
    outer = cuzeros(v, k, k, k)
    for i=1:k
        copyto!(view(outer, i, :, :), out1)
    end
    broadcast!(*, outer, v, outer)
    broadcast!(*, outer, -2/n, outer)
    for j=1:k
        v1 = view(outer,j, :, j)
        broadcast!(+, v1, v1, v)
        v1 = view(outer,j, j, :)
        broadcast!(+, v1, v1, v)
    end
    broadcast!(*, outer, 1/n, outer)
    return outer
end

function partial_derivative_outer(v::CuArray{T, 1}) where T
    k = length(v)
    out1 = v * v'
    n = v' * v
    outer = cuzeros(v, k, k, k)
    for i=1:k
        copyto!(view(outer, i, :, :), out1)
    end
    broadcast!(*, outer, v, outer)
    broadcast!(*, outer, -2/n, outer)
    for j=1:k
        v1 = view(outer,j, :, j)
        broadcast!(+, v1, v1, v)
        v1 = view(outer,j, j, :)
        broadcast!(+, v1, v1, v)
    end
    broadcast!(*, outer, 1/n, outer)
    return outer
end


# Everything the Householder weight gradients need from the data: the channel-by-channel
# Gram matrix `-2 * Σ_i X_i' * ΔY_i`, summed over samples and spatial positions. Each `∂V`
# slice then enters as a plain inner product against it, because
#
#     sum((-2 X_i * ∂V[m]) .* ΔY_i) = ⟨∂V[m], -2 X_i' ΔY_i⟩,
#
# so the contraction over the batch can happen once instead of once per `∂V` slice per
# sample. The former per-sample `Mat * Tens[i, :, :]` products rebuilt this same quantity one
# tiny matrix at a time and allocated far more than they computed.
function channel_gram(X::AbstractArray{T, N}, ΔY::AbstractArray{T, N}) where {T, N}
    k, batchsize = size(X, N-1), size(X, N)
    Xm, ΔYm = reshape(X, :, k, batchsize), reshape(ΔY, :, k, batchsize)
    G = cuzeros(X, k, k)
    @inbounds for i = 1:batchsize
        mul!(G, adjoint(view(Xm, :, :, i)), view(ΔYm, :, :, i), T(-2), one(T))
    end
    return G
end

# A loop of small GEMMs is the wrong shape for a GPU; batch them into one call instead.
function channel_gram(X::CuArray{T, N}, ΔY::CuArray{T, N}) where {T, N}
    k, batchsize = size(X, N-1), size(X, N)
    Xm, ΔYm = reshape(X, :, k, batchsize), reshape(ΔY, :, k, batchsize)
    G = NNlib.batched_mul(NNlib.batched_adjoint(Xm), ΔYm)
    return T(-2) .* dropdims(sum(G; dims=3); dims=3)
end

function conv1x1_grad_v(X::AbstractArray{T, N}, ΔY::AbstractArray{T, N},
                        C::Conv1x1; adjoint=false) where {T, N}

    # Reshape input
    v1 = C.v1.data
    v2 = C.v2.data
    v3 = C.v3.data
    k = length(v1)

    dv1 = cuzeros(X, k)
    dv2 = cuzeros(X, k)
    dv3 = cuzeros(X, k)

    # Do not calculate gradients if layer is frozen
    if C.freeze 
        return dv1, dv2, dv3 
    end

    V1 = v1*v1'/(v1'*v1)
    V2 = v2*v2'/(v2'*v2)
    V3 = v3*v3'/(v3'*v3)

    dV1 = partial_derivative_outer(v1)
    dV2 = partial_derivative_outer(v2)
    dV3 = partial_derivative_outer(v3)

    M1 = (I - 2 * (V2 + V3) + 4*V2*V3)
    M3 = (I - 2 * (V1 + V2) + 4*V1*V2)
    tmp = cuzeros(X, k, k)
    for i=1:k
        # dV1
        mul!(tmp, dV1[i, :, :], M1)
        @views copyto!(dV1[i, :, :], tmp)
        # dV2
        v2 = dV2[i, :, :]
        broadcast!(+, tmp, v2, 4 * V1 * v2 * V3 - 2 * (V1 * v2 + v2 * V3))
        @views copyto!(dV2[i, :, :], tmp)
        # dV3
        mul!(tmp, M3, dV3[i, :, :])
        @views copyto!(dV3[i, :, :], tmp)
    end

    # The `∂V` slices used to be transposed one at a time above; `⟨A', G⟩ == ⟨A, G'⟩` moves
    # that single transpose onto the k x k Gram matrix instead.
    G = channel_gram(X, ΔY)
    g = vec(adjoint ? permutedims(G) : G)
    mul!(dv1, reshape(dV1, k, k*k), g)
    mul!(dv2, reshape(dV2, k, k*k), g)
    mul!(dv3, reshape(dV3, k, k*k), g)
    return dv1, dv2, dv3
end


# Forward pass
#
# The log-determinant mode is resolved into a `Val` before the work starts, so that a caller
# passing `logdet=false` -- every coupling layer, on every pass -- gets a return shape
# inference can see. Reassigning a local `logdet` here instead left the return a three-way
# union that propagated into the callers' hot paths.
forward(X::AbstractArray{T, N}, C::Conv1x1; logdet=nothing) where {T, N} =
    _conv1x1_forward(X, C, logdet_mode(isnothing(logdet) ? C.logdet : logdet))

# The map is orthogonal, so its log-determinant is identically zero and a caller that only
# wants the mapping has nothing to gain from asking for it. Taking the mode as a `Val`
# argument makes the return type a property of the call rather than something constant
# propagation has to recover from a `logdet=false` keyword -- which it does not do.
conv1x1_forward(X::AbstractArray, C::Conv1x1) = _conv1x1_forward(X, C, Val(false))
conv1x1_inverse(Y::AbstractArray, C::Conv1x1) = _conv1x1_inverse(Y, C, Val(false))

function _conv1x1_forward(X::AbstractArray{T, N}, C::Conv1x1, mode::Val) where {T, N}
    Y = cuzeros(X, size(X)...)
    n_in = size(X, N-1)

    v1 = C.v1.data
    v2 = C.v2.data
    v3 = C.v3.data

    for i=1:size(X, N)
        Xi = reshape(selectdim(X, N, i), :, n_in)
        Yi = chain_lr(Xi, v1, v2, v3)
        selectdim(Y, N, i) .= reshape(Yi, size(selectdim(Y, N, i))...)
    end
    return _conv1x1_out(Y, X, mode)   # logdet always 0
end

# The map is orthogonal, so its log-determinant is zero -- but the shape still has to match
# whatever the rest of the chain is accumulating.
_conv1x1_out(Y, ::AbstractArray, ::Val{false}) = Y
_conv1x1_out(Y, ::AbstractArray{T, N}, ::Val{true}) where {T, N} = (Y, zero(T))
_conv1x1_out(Y, X::AbstractArray{T, N}, ::Val{:sample}) where {T, N} =
    (Y, constant_per_sample(X, zero(T)))

# Forward pass and update weights
function forward(X_tuple::Tuple, C::Conv1x1; set_grad::Bool=true)
    ΔX = X_tuple[1]
    X = X_tuple[2]
    ΔY = conv1x1_forward(ΔX, C)    # forward propagate residual
    Y = conv1x1_forward(X, C)      # recompute forward state
    Δv1, Δv2, Δv3 = conv1x1_grad_v(Y, ΔX, C; adjoint=true)  # gradient w.r.t. weights
    if set_grad
        isnothing(C.v1.grad) ? (C.v1.grad = Δv1) : (C.v1.grad += Δv1)
        isnothing(C.v2.grad) ? (C.v2.grad = Δv2) : (C.v2.grad += Δv2)
        isnothing(C.v3.grad) ? (C.v3.grad = Δv3) : (C.v3.grad += Δv3)
    else
        Δθ = [Parameter(Δv1), Parameter(Δv2), Parameter(Δv3)]
    end
    set_grad ? (return ΔY, Y) : (return ΔY, Δθ, Y)
end

# Inverse pass (see `forward` on why the mode is resolved before the work)
inverse(Y::AbstractArray{T, N}, C::Conv1x1; logdet=nothing) where {T, N} =
    _conv1x1_inverse(Y, C, logdet_mode(isnothing(logdet) ? C.logdet : logdet))

function _conv1x1_inverse(Y::AbstractArray{T, N}, C::Conv1x1, mode::Val) where {T, N}
    X = cuzeros(Y, size(Y)...)
    n_in = size(X, N-1)

    v1 = C.v1.data
    v2 = C.v2.data
    v3 = C.v3.data

    for i=1:size(Y, N)
        Yi = reshape(selectdim(Y, N, i), :, n_in)
        Xi = chain_lr(Yi, v3, v2, v1)
        selectdim(X, N, i) .= reshape(Xi, size(selectdim(X, N, i))...)
    end
    return _conv1x1_out(X, Y, mode)   # logdet always 0
end

# Inverse pass and update weights
function inverse(Y_tuple::Tuple, C::Conv1x1; set_grad::Bool=true)
    ΔY = Y_tuple[1]
    Y = Y_tuple[2]
    X = conv1x1_inverse(Y, C)  # recompute forward state
    if set_grad
        return inverse_grad(ΔY, X, C), X
    end
    ΔX, Δθ = inverse_grad(ΔY, X, C; set_grad=false)
    return ΔX, Δθ, X
end

"""
    ΔX, X = backward(ΔY, Y, C::Conv1x1)

 Backward pass in the array form every other layer implements, so that a `Conv1x1` can sit
 inside an [`InvertibleChain`](@ref) (or any other caller that backpropagates through a chain
 of layers with `backward(ΔY, Y, layer)`).

 Same computation as `C.inverse((ΔY, Y))`: the map is orthogonal, so the adjoint of the
 forward map is the inverse map. The Householder gradients are *accumulated* into `C.v1.grad`
 etc. rather than overwritten, as they have always been -- the caller is responsible for
 clearing them between passes, which is what `InvertibleChain`'s pullback does.

 `logdet_weight` is accepted for interface parity and validated like everywhere else, but has
 nothing to weight here: the log-determinant of an orthogonal map is identically zero and in
 particular does not depend on the Householder vectors.
"""
function backward(ΔY::AbstractArray{T, N}, Y::AbstractArray{T, N}, C::Conv1x1;
                  set_grad::Bool=true, logdet_weight=nothing) where {T, N}
    check_logdet_weight(logdet_weight, set_grad)
    return inverse((ΔY, Y), C; set_grad=set_grad)
end

## Reverse-layer functions
"""
    ΔY, Y = backward_inv(ΔX, X, C::Conv1x1)

 Backward pass of a reversed `Conv1x1`, which applies `inverse` in its forward direction: the
 cotangent is pushed through the forward map, and the Householder gradients accumulated as in
 [`backward`](@ref).
"""
function backward_inv(ΔX::AbstractArray{T, N}, X::AbstractArray{T, N}, C::Conv1x1;
                      set_grad::Bool=true, logdet_weight=nothing) where {T, N}
    check_logdet_weight(logdet_weight, set_grad)
    return forward((ΔX, X), C; set_grad=set_grad)
end

"""
    ΔX = inverse_grad(ΔY, X, C::Conv1x1)

 Backward pass of `inverse` for a caller that already holds `X = C.inverse(Y)` -- as the
 coupling layers do, having inverted `Y` on their own recomputation pass. `Y` is not inverted
 a second time; only the residual is propagated and the Householder gradients accumulated.
"""
inverse_grad(ΔY::AbstractArray{T, N}, X::AbstractArray{T, N}, C::Conv1x1;
             set_grad::Bool=true) where {T, N} =
    _inverse_grad(ΔY, X, C, Val(set_grad))

function _inverse_grad(ΔY::AbstractArray{T, N}, X::AbstractArray{T, N}, C::Conv1x1,
                       ::Val{grad}) where {T, N, grad}
    ΔX = conv1x1_inverse(ΔY, C)          # derivative w.r.t. input

    # Gradient w.r.t. weights
    # Will be zeros if layer is frozen (not learnable)
    Δv1, Δv2, Δv3 = conv1x1_grad_v(X, ΔY, C)
    if grad
        isnothing(C.v1.grad) ? (C.v1.grad = Δv1) : (C.v1.grad += Δv1)
        isnothing(C.v2.grad) ? (C.v2.grad = Δv2) : (C.v2.grad += Δv2)
        isnothing(C.v3.grad) ? (C.v3.grad = Δv3) : (C.v3.grad += Δv3)
        return ΔX
    end
    return ΔX, [Parameter(Δv1), Parameter(Δv2), Parameter(Δv3)]
end


## Jacobian-related functions

function jacobian(ΔX::AbstractArray{T, N}, Δθ::AbstractVector{<:Parameter}, X::AbstractArray{T, N}, C::Conv1x1) where {T, N}
    Y = cuzeros(X, size(X)...)
    ΔY = cuzeros(ΔX, size(ΔX)...)
    n_in = size(X, N-1)

    v1 = C.v1.data
    v2 = C.v2.data
    v3 = C.v3.data
    dv1 = Δθ[1].data
    dv2 = Δθ[2].data
    dv3 = Δθ[3].data

    for i=1:size(X, N)
        Xi = reshape(selectdim(X, N, i), :, n_in)
        isa(X, CUDA.CuArray) && (Xi = CUDA.CuArray(Xi))
        Yi = chain_lr(Xi, v1, v2, v3)
        selectdim(Y, N, i) .= reshape(Yi, size(selectdim(Y, N, i) )...)

        ΔXi = reshape(selectdim(ΔX, N, i), :, n_in)
        ΔYi = chain_lr(Xi, v1, v2, v3)
        # this is a lot of outer products of 1D vecotrs, need to be cleaned up that's overkill computationnaly
        n1 = norm(v1); n2 = norm(v2); n3 = norm(v3);
        c1 = I - 2f0*v1*v1'/n1^2f0; c2 = I - 2f0*v2*v2'/n2^2f0; c3 = I - 2f0*v3*v3'/n3^2f0;
        ΔYi = chain_lr(ΔXi, v1, v2, v3)
        ΔYi += -2f0*Xi*((dv1*v1'+v1*dv1'-2f0*dot(v1,dv1)*v1*v1'/n1^2f0)/n1^2f0*c2*c3+
                       c1*(dv2*v2'+v2*dv2'-2f0*dot(v2,dv2)*v2*v2'/n2^2f0)/n2^2f0*c3+
                       c1*c2*(dv3*v3'+v3*dv3'-2f0*dot(v3,dv3)*v3*v3'/n3^2f0)/n3^2f0)
        selectdim(ΔY, N, i) .= reshape(ΔYi, size(selectdim(ΔY, N, i))...)
    end

    return ΔY, Y
end

function adjointJacobian(ΔY::AbstractArray{T, N}, Y::AbstractArray{T, N}, C::Conv1x1) where {T, N}
    return inverse((ΔY, Y), C; set_grad=false)
end

function jacobianInverse(ΔY::AbstractArray{T, N}, Δθ::AbstractVector{<:Parameter}, Y::AbstractArray{T, N}, C::Conv1x1) where {T, N}
    return inverse(C).jacobian(ΔY, Δθ[end:-1:1], Y)
end

function adjointJacobianInverse(ΔX::AbstractArray{T, N}, X::AbstractArray{T, N}, C::Conv1x1) where {T, N}
    ΔX, Δθinv, X = inverse(C).adjointJacobian(ΔX, X)
    return ΔX, Δθinv[end:-1:1], X
end

function inverse(C::Conv1x1)
    return Conv1x1(C.k, C.v3, C.v2, C.v1, C.logdet, C.freeze)
end
