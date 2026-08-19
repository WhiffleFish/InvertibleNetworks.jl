# Dense (non-convolutional) conditioner block for coupling layers.
#
# The convolutional `ResidualBlock` is the right conditioner for image data and the wrong one
# for vectors. A flow on `(dim, batch)` data has no spatial extent, so every convolution in
# it is a 1x1 convolution: a GEMM routed through the convolution machinery, which on the GPU
# means a cuDNN dispatch (and its algorithm selection) for something BLAS does directly.
# This block is the same three-layer shape, written in terms of matrix multiplies.

export MLPBlock, ConditionerBlock

"""
    B = MLPBlock(d_in, n_hidden; d_out=2*d_in, activation=ReLUlayer(), fan=false)

 Create a (non-invertible) dense conditioner block: a three-layer perceptron with a skip
 connection around the middle layer, shaped so that it can stand in for
 [`ResidualBlock`](@ref) inside a coupling layer.

 The block flattens everything but the trailing batch dimension, so it accepts an input of
 any rank: `(d_in, batch)` vectors directly, or `(dims..., channels, batch)` tensors with
 `d_in = prod(dims)*channels`. The output is reshaped back to the input's leading dimensions,
 carrying `d_out ÷ prod(dims)` channels.

 *Input*:

 - `d_in`: number of input features, i.e. the flattened size of one sample

 - `n_hidden`: width of the two hidden layers

 - `d_out`: number of output features (default `2*d_in`)

 - `activation`: activation between layers

 - `fan`: `true` gives a linear output of `d_out` features, which is what a coupling layer's
    concatenated (log-scale, shift) output has to be. `false` applies a gated linear unit,
    halving `d_out`.

 - `out_scale`: factor applied to the initialization of the output weight `W3`; see
    [`OUT_INIT_SCALE`](@ref).

 *Output*:

 - `B`: dense block

 *Usage:*

 - Forward mode: `Y = B.forward(X)`

 - Backward mode: `ΔX = B.backward(ΔY, X)`

 *Trainable parameters:*

 - Weights `B.W1`, `B.W2`, `B.W3` and biases `B.b1`, `B.b2`

 See also: [`ResidualBlock`](@ref), [`CouplingLayerGlow`](@ref), [`get_params`](@ref),
 [`clear_grad!`](@ref)
"""
struct MLPBlock{PW<:Parameter,PB<:Parameter,A<:ActivationFunction} <: NeuralNetLayer
    W1::PW
    W2::PW
    W3::PW
    b1::PB
    b2::PB
    fan::Bool
    activation::A
end

Flux.@layer MLPBlock

#######################################################################################################################
#  Constructors

function MLPBlock(d_in::Integer, n_hidden::Integer; d_out=nothing,
                  activation::ActivationFunction=ReLUlayer(), fan::Bool=false,
                  out_scale=OUT_INIT_SCALE)
    isnothing(d_out) && (d_out = 2*d_in)
    W1 = Parameter(glorot_uniform(n_hidden, d_in))
    W2 = Parameter(glorot_uniform(n_hidden, n_hidden))
    W3 = Parameter(out_scale .* glorot_uniform(d_out, n_hidden))
    b1 = Parameter(zeros(Float32, n_hidden))
    b2 = Parameter(zeros(Float32, n_hidden))
    return MLPBlock(W1, W2, W3, b1, b2, fan, activation)
end

# Constructor for given weights
MLPBlock(W1, W2, W3, b1, b2; activation::ActivationFunction=ReLUlayer(), fan::Bool=false) =
    MLPBlock(Parameter(W1), Parameter(W2), Parameter(W3), Parameter(b1), Parameter(b2),
             fan, activation)

#######################################################################################################################
# Flattening
#
# Only the trailing batch dimension is meaningful to a dense layer. For the `(dim, batch)`
# case `size(X)[1:N-2]` is empty and both of these are the identity, so a vector flow pays
# nothing for them.

@inline _flat(X::AbstractArray{T, N}) where {T, N} = reshape(X, :, size(X, N))

@inline function _unflat(Yf::AbstractMatrix, X::AbstractArray{T, N}) where {T, N}
    lead = ntuple(i -> size(X, i), Val(N-2))
    return reshape(Yf, lead..., size(Yf, 1) ÷ prod(lead), size(Yf, 2))
end

# `W*X .+ b` as a single GEMM: the bias is written into the destination and the multiply
# accumulates onto it (`beta = 1`), rather than allocating the product and then broadcasting
# the bias across it.
@inline function _affine!(Y::AbstractMatrix{T}, W, X, b) where T
    Y .= b
    mul!(Y, W, X, one(T), one(T))
    return Y
end

# As above, with the middle layer's skip connection folded into the same accumulation.
@inline function _affine_skip!(Y::AbstractMatrix{T}, W, X, b) where T
    Y .= X .+ b
    mul!(Y, W, X, one(T), one(T))
    return Y
end

# In-place activation where a scalar form is available: on the non-saving path the
# pre-activation is dead as soon as it is consumed, so allocating a second array for the
# result buys nothing.
@inline _activate!(a::ActivationFunction, Y) = __activate!(scalar_form(a.forward), a, Y)
@inline __activate!(::Nothing, a::ActivationFunction, Y) = a.forward(Y)
@inline __activate!(f, ::ActivationFunction, Y) = (Y .= f.(Y); Y)

#######################################################################################################################
# Functions

forward(X::AbstractArray{T, N}, B::MLPBlock; save=false) where {T, N} =
    _forward(X, B, Val(save))

block_forward(X, B::MLPBlock) = _forward(X, B, Val(false))
block_forward_save(X, B::MLPBlock) = _forward(X, B, Val(true))

# The saved state carries the flat working matrices, plus the block's output in the caller's
# own shape so that the generic `block_output` hands back something a coupling layer can
# broadcast against its data.
function _forward(X::AbstractArray{T, N}, B::MLPBlock, ::Val{save}) where {T, N, save}
    Xf = _flat(X)
    nb = size(Xf, 2)
    # Without this, a mismatch surfaces as a BLAS dimension error several frames down.
    size(Xf, 1) == size(B.W1.data, 2) || throw(DimensionMismatch(
        "MLPBlock expects $(size(B.W1.data, 2)) features per sample, but an input of size " *
        "$(size(X)) flattens to $(size(Xf, 1))"))

    Y1 = _affine!(similar(Xf, size(B.W1.data, 1), nb), B.W1.data, Xf, B.b1.data)
    X2 = save ? B.activation.forward(Y1) : _activate!(B.activation, Y1)

    Y2 = _affine_skip!(similar(X2), B.W2.data, X2, B.b2.data)
    X3 = save ? B.activation.forward(Y2) : _activate!(B.activation, Y2)

    Y3 = B.W3.data * X3
    # `fan=true` is the coupling-conditioner output and must stay linear: an activation here
    # would confine the log-scale and the shift to one sign.
    X4 = B.fan ? Y3 : GaLU(Y3)

    save && (return Y1, Y2, Y3, X2, X3, _unflat(X4, X))
    return _unflat(X4, X)
end

# Backward
backward(ΔX4::AbstractArray{T, N}, X::AbstractArray{T, N}, B::MLPBlock;
         set_grad::Bool=true) where {T, N} =
    block_backward(ΔX4, X, block_forward_save(X, B), B; set_grad=set_grad)

# Backward from already-recomputed forward states, as produced by `block_forward_save`.
function block_backward(ΔX4::AbstractArray{T, N}, X::AbstractArray{T, N}, state::Tuple,
                        B::MLPBlock; set_grad::Bool=true) where {T, N}
    Y1, Y2, Y3, X2, X3, _ = state
    Xf = _flat(X)

    # Everything below runs on the flat matrices. `GaLU` splits the leading dimension, which
    # is exactly the flattened feature axis, so it needs no reshaping either.
    ΔY3 = B.fan ? _flat(ΔX4) : GaLUgrad(_flat(ΔX4), Y3)

    ΔW3 = ΔY3 * adjoint(X3)
    ΔX3 = adjoint(B.W3.data) * ΔY3

    ΔY2 = backward(ΔX3, Y2, X3, B.activation)
    ΔX2 = copy(ΔY2)                                  # the middle layer's skip connection
    mul!(ΔX2, adjoint(B.W2.data), ΔY2, one(T), one(T))
    ΔW2 = ΔY2 * adjoint(X2)
    Δb2 = dropdims(sum(ΔY2; dims=2); dims=2)

    ΔY1 = backward(ΔX2, Y1, X2, B.activation)
    ΔW1 = ΔY1 * adjoint(Xf)
    Δb1 = dropdims(sum(ΔY1; dims=2); dims=2)
    ΔX = reshape(adjoint(B.W1.data) * ΔY1, size(X))

    if set_grad
        B.W1.grad = ΔW1
        B.W2.grad = ΔW2
        B.W3.grad = ΔW3
        B.b1.grad = Δb1
        B.b2.grad = Δb2
        return ΔX
    else
        return ΔX, [Parameter(ΔW1), Parameter(ΔW2), Parameter(ΔW3),
                    Parameter(Δb1), Parameter(Δb2)]
    end
end

## Jacobian-related functions

jacobian(::AbstractArray{T, N}, ::AbstractVector{<:Parameter}, ::AbstractArray{T, N},
         ::MLPBlock) where {T, N} =
    throw(ArgumentError("Jacobian for MLPBlock not implemented; use the adjoint Jacobian, " *
                        "or a ResidualBlock conditioner, for the Jacobian interface"))

adjointJacobian(ΔX4::AbstractArray{T, N}, X::AbstractArray{T, N}, B::MLPBlock) where {T, N} =
    backward(ΔX4, X, B; set_grad=false)


#######################################################################################################################

"""
    ConditionerBlock

 The non-invertible blocks a coupling layer can use to predict its coupling coefficients:
 [`ResidualBlock`](@ref) (convolutional), [`MLPBlock`](@ref) (dense) or [`FluxBlock`](@ref)
 (an arbitrary Flux `Chain`).
"""
const ConditionerBlock = Union{ResidualBlock,FluxBlock,MLPBlock}

# A coupling layer reads its conditioner's output as a concatenated (log-scale, shift) pair,
# which is what `fan=true` produces. A `FluxBlock` is an arbitrary chain with no such flag, so
# there is nothing to check for it.
_check_fan(::FluxBlock) = nothing
_check_fan(B) = B.fan || throw(ArgumentError(
    "$(nameof(typeof(B))) must be built with fan=true so that it emits the concatenated " *
    "(log-scale, shift) pair a coupling layer needs"))
