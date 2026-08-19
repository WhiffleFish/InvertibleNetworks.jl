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

# The cotangent of the `k x k` matrix the layer applies: `Q̄ = Σ_i X_i' * ΔY_i`, summed over
# samples and spatial positions.
#
# `Y[s, c', b] = Σ_c X[s, c, b] * Q[c, c']`, so `∂L/∂Q[c, c'] = Σ_{s,b} X[s, c, b] ΔY[s, c', b]`
# -- exactly this Gram matrix. It is the *only* place the weight gradient touches the data:
# everything downstream of it is `k x k` algebra with no batch dimension, which is what lets
# `householder_grad` do the rest at a cost independent of the batch size.
function channel_gram(X::AbstractArray{T, N}, ΔY::AbstractArray{T, N}) where {T, N}
    k, batchsize = size(X, N-1), size(X, N)
    Xm, ΔYm = reshape(X, :, k, batchsize), reshape(ΔY, :, k, batchsize)
    G = cuzeros(X, k, k)
    @inbounds for i = 1:batchsize
        mul!(G, adjoint(view(Xm, :, :, i)), view(ΔYm, :, :, i), one(T), one(T))
    end
    return G
end

# A loop of small GEMMs is the wrong shape for a GPU; batch them into one call instead.
function channel_gram(X::CuArray{T, N}, ΔY::CuArray{T, N}) where {T, N}
    k, batchsize = size(X, N-1), size(X, N)
    Xm, ΔYm = reshape(X, :, k, batchsize), reshape(ΔY, :, k, batchsize)
    G = NNlib.batched_mul(NNlib.batched_adjoint(Xm), ΔYm)
    return dropdims(sum(G; dims=3); dims=3)
end

# Gradient of the loss with respect to the three Householder vectors.
#
# The layer applies one `k x k` matrix `Q = H(v1)H(v2)H(v3)` to every sample, so this gradient
# factors cleanly in two: `channel_gram` contracts the data down to `Q̄ = ∂L/∂Q`, and
# `householder_grad` pulls that back through `v... -> Q`. Only the first piece sees the batch;
# the second is `k x k` algebra whose cost does not depend on how much data went in.
#
# The previous version instead materialized `∂V_i/∂v_i` as a `k x k x k` tensor for each
# reflector and hit every one of its `k` slices with a `k x k` matrix multiply -- `O(k^4)` work
# and `O(k^3)` memory, recomputed identically on every backward pass no matter the batch size.
# For a vector flow at `k = 128` that term alone was ~190x the cost of the entire rest of the
# gradient.
#
# `adjoint=true` is for the reversed direction, which applies `Q'`: the cotangent of `Q` is then
# the transpose of the one the forward direction would report.
function conv1x1_grad_v(X::AbstractArray{T, N}, ΔY::AbstractArray{T, N},
                        C::Conv1x1; adjoint=false) where {T, N}
    k = length(C.v1.data)

    # Do not calculate gradients if layer is frozen (not learnable)
    C.freeze && return cuzeros(X, k), cuzeros(X, k), cuzeros(X, k)

    G = channel_gram(X, ΔY)
    Q̄ = adjoint ? permutedims(G) : G
    return householder_grad(Q̄, C.v1.data, C.v2.data, C.v3.data)
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

# The Householder chain is one orthogonal matrix, and it is the same one for every sample, so
# the whole batch is a single contraction against it. Applying the reflectors sample by sample
# -- as this did -- turned each pass into `batchsize` rank-1 BLAS-2 updates on a `1 x k` slice,
# which is why the per-sample cost used to be flat in the batch size instead of amortizing.
function _conv1x1_forward(X::AbstractArray{T, N}, C::Conv1x1, mode::Val) where {T, N}
    Q = householder_matrix(C.v1.data, C.v2.data, C.v3.data)
    Y = apply_channel_matrix(X, Q)
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

# Reflectors are their own inverses, so the reversed chain is the transpose of the forward
# matrix; it is built from the reversed vectors here to keep the correspondence with
# `_conv1x1_forward` visible rather than implied.
function _conv1x1_inverse(Y::AbstractArray{T, N}, C::Conv1x1, mode::Val) where {T, N}
    Qinv = householder_matrix(C.v3.data, C.v2.data, C.v1.data)
    X = apply_channel_matrix(Y, Qinv)
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

# Directional derivative of the reflector `I - 2vv'/(v'v)` with respect to `v`, in the
# direction `dv`, with the leading `-2` left to the caller.
function householder_differential(v::AbstractVector{T}, dv::AbstractVector{T}) where T
    n = dot(v, v)
    return (dv*adjoint(v) + v*adjoint(dv) - (2*dot(v, dv)/n)*(v*adjoint(v))) / n
end

# Both the map and its derivative are contractions of the batch against a `k x k` matrix that
# does not depend on the sample, so the whole Householder algebra -- three reflectors, three
# differentials and their products -- happens once per call. It used to be rebuilt from
# scratch on every iteration of a loop over the batch, which cost `batchsize` times more than
# the data movement it was wrapped around.
function jacobian(ΔX::AbstractArray{T, N}, Δθ::AbstractVector{<:Parameter}, X::AbstractArray{T, N}, C::Conv1x1) where {T, N}
    v1, v2, v3 = C.v1.data, C.v2.data, C.v3.data
    dv1, dv2, dv3 = Δθ[1].data, Δθ[2].data, Δθ[3].data

    Q = householder_matrix(v1, v2, v3)
    Y = apply_channel_matrix(X, Q)

    H1 = householder_reflector(v1)
    H2 = householder_reflector(v2)
    H3 = householder_reflector(v3)
    dH1 = householder_differential(v1, dv1)
    dH2 = householder_differential(v2, dv2)
    dH3 = householder_differential(v3, dv3)

    # d/dθ of `Q = H1*H2*H3`, by the product rule.
    dQ = T(-2) .* (dH1*H2*H3 .+ H1*dH2*H3 .+ H1*H2*dH3)

    ΔY = apply_channel_matrix(ΔX, Q) .+ apply_channel_matrix(X, dQ)

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
