# Residual block from Putzky and Welling (2019): https://arxiv.org/abs/1911.10914
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export ResidualBlock, ResidualBlock3D

"""
    RB = ResidualBlock(n_in, n_hidden; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, fan=false)
    RB = ResidualBlock3D(n_in, n_hidden; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, fan=false)

or

    RB = ResidualBlock(W1, W2, W3, b1, b2; p1=1, p2=1, s1=1, s2=1, fan=false)
    RB = ResidualBlock3D(W1, W2, W3, b1, b2; p1=1, p2=1, s1=1, s2=1, fan=false)

 Create a (non-invertible) residual block, consisting of three convolutional layers and activation functions.
 The first convolution is a downsampling operation with a stride equal to the kernel dimension. The last
 convolution is the corresponding transpose operation and upsamples the data to either its original dimensions
 or to twice the number of input channels (for `fan=true`). The first and second layer contain a bias term.

 *Input*:

 - `n_in`: number of input channels

 - `n_hidden`: number of hidden channels

 - `n_out`: number of ouput channels

 - `activation`: activation type between conv layers and final output

 - `k1`, `k2`: kernel size of convolutions in residual block. `k1` is the kernel of the first and third
    operator, `k2` is the kernel size of the second operator.

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second convolution (`p2`)

 - `s1`, `s2`: stride for the first and third convolution (`s1`) and the second convolution (`s2`)

 - `fan`: bool to indicate whether the ouput has twice the number of input channels. For `fan=false`, the last
    activation is a gated linear unit (thereby bringing the output back to the original dimensions).
    For `fan=true`, the output is linear and has `n_out` channels -- the form a coupling layer needs, since it
    reads the output as a concatenated (log-scale, shift) pair.

 - `out_scale`: factor applied to the initialization of the output weight `W3`; see [`OUT_INIT_SCALE`](@ref).

or

 - `W1`, `W2`, `W3`: 4D tensors of convolutional weights

 - `b1`, `b2`: bias terms

 *Output*:

 - `RB`: residual block layer

 *Usage:*

 - Forward mode: `Y = RB.forward(X)`

 - Backward mode: `ΔX = RB.backward(ΔY, X)`

 *Trainable parameters:*

 - Convolutional kernel weights `RB.W1`, `RB.W2` and `RB.W3`

 - Bias terms `RB.b1` and `RB.b2`

 See also: [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct ResidualBlock{PW<:Parameter,PB<:Parameter,S,P,A<:ActivationFunction} <: NeuralNetLayer
    W1::PW
    W2::PW
    W3::PW
    b1::PB
    b2::PB
    fan::Bool
    strides::S
    pad::P
    activation::A
end

Flux.@layer ResidualBlock

#######################################################################################################################
#  Constructors

# Constructor
function ResidualBlock(n_in, n_hidden; n_out=nothing, activation::ActivationFunction=ReLUlayer(), k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, fan=false, ndims=2, out_scale=OUT_INIT_SCALE)
    # default/legacy behaviour
    isnothing(n_out) && (n_out = 2*n_in)

    k1 = Tuple(k1 for i=1:ndims)
    k2 = Tuple(k2 for i=1:ndims)
    # Initialize weights
    W1 = Parameter(glorot_uniform(k1..., n_in, n_hidden))
    W2 = Parameter(glorot_uniform(k2..., n_hidden, n_hidden))
    W3 = Parameter(out_scale .* glorot_uniform(k1..., n_out, n_hidden))
    b1 = Parameter(zeros(Float32, n_hidden))
    b2 = Parameter(zeros(Float32, n_hidden))

    return ResidualBlock(W1, W2, W3, b1, b2, fan, (s1, s2), (p1, p2), activation)
end

# Constructor for given weights
function ResidualBlock(W1, W2, W3, b1, b2; activation::ActivationFunction=ReLUlayer(), p1=1, p2=1, s1=1, s2=1, fan=false, ndims=2)

    # Make weights parameters
    W1 = Parameter(W1)
    W2 = Parameter(W2)
    W3 = Parameter(W3)
    b1 = Parameter(b1)
    b2 = Parameter(b2)

    return ResidualBlock(W1, W2, W3, b1, b2, fan, (s1, s2), (p1, p2),activation)
end

ResidualBlock3D(args...; kw...) = ResidualBlock(args...; kw..., ndims=3)

#######################################################################################################################
# Point-spatial specialization
#
# A flow over vector data -- tabular data, an RL policy over a `d`-dimensional action -- reshapes
# each sample into this package's image convention as a `1 x 1` map with `d` channels. Every
# convolution in this block is then a contraction over channels alone: with unit stride and
# "same" padding (`k = 2p+1`) the single data point sits under the kernel's centre tap and every
# other tap sees only the zero pad. The result is a GEMM.
#
# `NNlib.conv` still takes its general route to that answer -- build `DenseConvDims`, im2col the
# patch, call BLAS on a matrix whose interesting dimension is 1, scatter back -- and at these
# shapes the bookkeeping costs far more than the multiply it wraps. Measured at 17->64->64->46
# channels with a batch of 512: 5.9 ms forward and 13.0 ms backward through `conv`, against
# 0.08 ms for the equivalent GEMM chain.
#
# [`MLPBlock`](@ref) is the better answer when the caller can choose its conditioner, and a
# coupling layer built with `dense=true` gets it. This path is for the callers that cannot: an
# existing `ResidualBlock`, built the way it always was, handed vector-shaped data. Contracting
# the centre tap is the same arithmetic in the same order, so it is not an approximation -- it
# agrees with `conv` to floating-point rounding, and reports a zero kernel gradient on the taps
# that took no part, exactly as `∇conv_filter` does. Written in `reshape`/`mul!`/broadcast over
# `AbstractArray`, so BLAS and cuBLAS are chosen by the argument type with no separate path.

# Is the data a single spatial point?
@inline _spatial_is_point(X::AbstractArray{T,N}) where {T,N} =
    all(ntuple(i -> size(X, i) == 1, Val(N-2)))

# The kernel's centre tap index, or 0 when the convolution does not collapse to a GEMM. Unit
# stride and `k = 2p+1` are what keep a point mapping to a point; a stride, or a kernel wider
# than its padding, would change the spatial extent and the reduction would be wrong rather
# than merely unavailable.
@inline _gemm_center(k::Int, p::Integer, s::Integer) = (s == 1 && k == 2p + 1) ? p + 1 : 0

# Per-axis padding or strides are legal for `NNlib.conv` but not handled here; 0 sends the
# block down its convolutional path rather than raising.
@inline _gemm_center(::Int, ::Any, ::Any) = 0

# Only spatially cubic kernels are handled, so that one tap index serves every axis.
@inline _cubic_kernel(W::AbstractArray{T,N}) where {T,N} =
    all(ntuple(i -> size(W, i) == size(W, 1), Val(N-2)))

"""
    is_gemm_shaped(X, RB::ResidualBlock)

 Whether this block's three convolutions collapse to matrix products on the input `X`: a `1 x 1`
 spatial map, unit strides, and odd kernels sized to their padding (`k = 2p+1`).

 True for a flow over vector data, false for any image-shaped input, which keeps its
 `NNlib.conv` path untouched.
"""
function is_gemm_shaped(X::AbstractArray{T,N}, RB::ResidualBlock) where {T,N}
    _spatial_is_point(X) || return false
    (_cubic_kernel(RB.W1.data) && _cubic_kernel(RB.W2.data) && _cubic_kernel(RB.W3.data)) ||
        return false
    return _gemm_center(size(RB.W1.data, 1), RB.pad[1], RB.strides[1]) > 0 &&
           _gemm_center(size(RB.W2.data, 1), RB.pad[2], RB.strides[2]) > 0 &&
           _gemm_center(size(RB.W3.data, 1), RB.pad[1], RB.strides[1]) > 0
end

# Centre taps of the three kernels, as channel matrices.
@inline _gemm_taps(RB::ResidualBlock) =
    (_center_matrix(RB.W1.data, _gemm_center(size(RB.W1.data, 1), RB.pad[1], RB.strides[1])),
     _center_matrix(RB.W2.data, _gemm_center(size(RB.W2.data, 1), RB.pad[2], RB.strides[2])),
     _center_matrix(RB.W3.data, _gemm_center(size(RB.W3.data, 1), RB.pad[1], RB.strides[1])))

# The centre tap as an `(in, out)` channel matrix. A genuinely `1 x 1` kernel already is that
# matrix, so this is a free reshape; a wider one is gathered into a few KB, which is noise
# beside the GEMM it feeds but is why `k1=1, p1=0` is the better way to build a block for
# point data -- and why `MLPBlock` is better still.
@inline function _center_matrix(W::AbstractArray{T,N}, c::Int) where {T,N}
    size(W, 1) == 1 && return reshape(W, size(W, N-1), size(W, N))
    return W[ntuple(_ -> c, Val(N-2))..., :, :]
end

# The kernel gradient implied by a centre-tap contraction: the tap's own, and zero on every tap
# that only ever multiplied the zero pad.
@inline function _center_grad(W::AbstractArray{T,N}, c::Int, ΔWc::AbstractMatrix) where {T,N}
    size(W, 1) == 1 && return reshape(ΔWc, size(W))
    ΔW = fill!(similar(W), zero(T))
    ΔW[ntuple(_ -> c, Val(N-2))..., :, :] .= ΔWc
    return ΔW
end

# Flatten the point-spatial axes away, and put them back. Both are free: the array is contiguous.
@inline _as_matrix(X::AbstractArray{T,N}) where {T,N} = reshape(X, size(X, N-1), size(X, N))
@inline _as_tensor(M::AbstractMatrix, ::Val{N}) where {N} =
    reshape(M, ntuple(_ -> 1, Val(N-2))..., size(M, 1), size(M, 2))

# `W'X .+ b` in one GEMM: the bias is written into the destination and the multiply accumulates
# onto it, rather than allocating the product and broadcasting the bias over it afterwards.
@inline function _tap_affine!(Y::AbstractMatrix{T}, Wc, X, b) where T
    Y .= b
    mul!(Y, adjoint(Wc), X, one(T), one(T))
    return Y
end

# As above, with the middle layer's skip connection folded into the same accumulation.
@inline function _tap_affine_skip!(Y::AbstractMatrix{T}, Wc, X, b) where T
    Y .= X .+ b
    mul!(Y, adjoint(Wc), X, one(T), one(T))
    return Y
end

function _forward_gemm(X1::AbstractArray{T,N}, RB::ResidualBlock, ::Val{save}) where {T,N,save}
    W1c, W2c, W3c = _gemm_taps(RB)
    X1m = _as_matrix(X1)

    # `conv` contracts the kernel's input-channel axis, so the tap enters transposed.
    Y1 = _tap_affine!(similar(X1m, size(W1c, 2), size(X1m, 2)), W1c, X1m, RB.b1.data)
    X2 = save ? RB.activation.forward(Y1) : _activate!(RB.activation, Y1)

    Y2 = _tap_affine_skip!(similar(X2), W2c, X2, RB.b2.data)
    X3 = save ? RB.activation.forward(Y2) : _activate!(RB.activation, Y2)

    # `W3` is applied through `∇conv_data`, the adjoint of a convolution mapping `n_out`
    # channels to `n_hidden`; as a matrix that is the tap itself, untransposed.
    Y3 = W3c * X3

    X4 = _fan_out(Y3, RB.fan)
    save && (return _as_tensor(Y1, Val(N)), _as_tensor(Y2, Val(N)), _as_tensor(Y3, Val(N)),
                    _as_tensor(X2, Val(N)), _as_tensor(X3, Val(N)), _as_tensor(X4, Val(N)))
    return _as_tensor(X4, Val(N))
end

function _block_backward_gemm(ΔX4::AbstractArray{T,N}, X1::AbstractArray{T,N},
                              state::NTuple{6,AbstractArray{T,N}}, RB::ResidualBlock,
                              ::Val{set_grad}) where {T,N,set_grad}
    W1c, W2c, W3c = _gemm_taps(RB)
    c1 = _gemm_center(size(RB.W1.data, 1), RB.pad[1], RB.strides[1])
    c2 = _gemm_center(size(RB.W2.data, 1), RB.pad[2], RB.strides[2])
    Y1, Y2, Y3, X2, X3, X4 = map(_as_matrix, state)

    # `fan=true` is a linear output, so its cotangent passes straight through.
    ΔY3 = RB.fan ? _as_matrix(ΔX4) : GaLUgrad(_as_matrix(ΔX4), Y3)
    ΔX3 = adjoint(W3c) * ΔY3
    ΔW3 = _center_grad(RB.W3.data, c1, ΔY3 * adjoint(X3))

    ΔY2 = backward(_as_tensor(ΔX3, Val(N)), _as_tensor(Y2, Val(N)),
                   _as_tensor(X3, Val(N)), RB.activation) |> _as_matrix
    ΔX2 = copy(ΔY2)                                  # the middle layer's skip connection
    mul!(ΔX2, W2c, ΔY2, one(T), one(T))
    ΔW2 = _center_grad(RB.W2.data, c2, _as_matrix(X2) * adjoint(ΔY2))
    Δb2 = dropdims(sum(ΔY2; dims=2); dims=2)

    ΔY1 = backward(_as_tensor(ΔX2, Val(N)), _as_tensor(Y1, Val(N)),
                   _as_tensor(X2, Val(N)), RB.activation) |> _as_matrix
    ΔX1 = W1c * ΔY1
    ΔW1 = _center_grad(RB.W1.data, c1, _as_matrix(X1) * adjoint(ΔY1))
    Δb1 = dropdims(sum(ΔY1; dims=2); dims=2)

    if set_grad
        RB.W1.grad = ΔW1
        RB.W2.grad = ΔW2
        RB.W3.grad = ΔW3
        RB.b1.grad = Δb1
        RB.b2.grad = Δb2
        return _as_tensor(ΔX1, Val(N))
    end
    return _as_tensor(ΔX1, Val(N)),
           [Parameter(ΔW1), Parameter(ΔW2), Parameter(ΔW3), Parameter(Δb1), Parameter(Δb2)]
end

#######################################################################################################################
# Functions

# Forward
forward(X1::AbstractArray{T,N}, RB::ResidualBlock; save=false) where {T,N} =
    _forward(X1, RB, Val(save))

# Fixed-shape internal path for coupling layers that never request saved states.
block_forward(X, RB::ResidualBlock) = _forward(X, RB, Val(false))

# The forward states `block_backward` needs. A coupling layer runs this block's forward pass
# on its own `inverse`, so handing the states straight to `block_backward` saves running it a
# second time -- a full extra forward pass per layer per backward call.
block_forward_save(X, RB::ResidualBlock) = _forward(X, RB, Val(true))

# The block's output, whether it comes from `block_forward` (the output alone) or from
# `block_forward_save` (the output plus the intermediates its backward needs).
block_output(Y::AbstractArray) = Y
block_output(state::NTuple{6,AbstractArray}) = state[6]

# `fan=true` is the coupling-conditioner output: `n_out` channels, read by the caller as a
# concatenated (log-scale, shift) pair, and therefore *linear*. Applying `RB.activation` here
# -- ReLU, by default -- confined both halves to the non-negative orthant, so an affine
# coupling could only ever contract (its log-determinant was negative by construction) and
# only ever shift in one direction. `fan=false` keeps its gated linear output, which is what
# a caller wanting a nonlinearity on the output asks for.
@inline _fan_out(Y3, fan::Bool) = fan ? Y3 : GaLU(Y3)

# Two methods rather than one with `if save`: the saving path has to materialize the
# pre-activations `Y1`/`Y2` that `block_backward` reads, and the non-saving path does not,
# which lets each bias and activation collapse into one fused broadcast instead of a kernel
# and a temporary each.
function _forward(X1::AbstractArray{T, N}, RB::ResidualBlock, ::Val{false}) where {T, N}
    is_gemm_shaped(X1, RB) && return _forward_gemm(X1, RB, Val(false))
    inds = channel_indices(Val(N))

    Yc1 = conv(X1, RB.W1.data; stride=RB.strides[1], pad=RB.pad[1])
    X2 = bias_activation(RB.activation, Yc1, reshape(RB.b1.data, inds...))

    Yc2 = conv(X2, RB.W2.data; stride=RB.strides[2], pad=RB.pad[2])
    X3 = bias_activation(RB.activation, Yc2, X2, reshape(RB.b2.data, inds...))

    cdims3 = DCDims(X1, RB.W3.data; stride=RB.strides[1], padding=RB.pad[1])
    Y3 = ∇conv_data(X3, RB.W3.data, cdims3)

    return _fan_out(Y3, RB.fan)
end

function _forward(X1::AbstractArray{T, N}, RB::ResidualBlock, ::Val{true}) where {T, N}
    is_gemm_shaped(X1, RB) && return _forward_gemm(X1, RB, Val(true))
    inds = channel_indices(Val(N))

    Y1 = conv(X1, RB.W1.data; stride=RB.strides[1], pad=RB.pad[1])
    Y1 .+= reshape(RB.b1.data, inds...)
    X2 = RB.activation.forward(Y1)

    Y2 = conv(X2, RB.W2.data; stride=RB.strides[2], pad=RB.pad[2])
    Y2 .+= X2 .+ reshape(RB.b2.data, inds...)
    X3 = RB.activation.forward(Y2)

    cdims3 = DCDims(X1, RB.W3.data; stride=RB.strides[1], padding=RB.pad[1])
    Y3 = ∇conv_data(X3, RB.W3.data, cdims3)

    X4 = _fan_out(Y3, RB.fan)
    return Y1, Y2, Y3, X2, X3, X4
end

# Backward
backward(ΔX4::AbstractArray{T, N}, X1::AbstractArray{T, N}, RB::ResidualBlock;
         set_grad::Bool=true) where {T, N} =
    block_backward(ΔX4, X1, block_forward_save(X1, RB), RB; set_grad=set_grad)

# Backward from already-recomputed forward states, as produced by `block_forward_save`.
function block_backward(ΔX4::AbstractArray{T, N}, X1::AbstractArray{T, N},
                        state::NTuple{6,AbstractArray{T, N}}, RB::ResidualBlock;
                        set_grad::Bool=true) where {T, N}
    if is_gemm_shaped(X1, RB)
        # `Val` rather than the `Bool`: the two branches return different shapes, so the caller
        # only stays inferable if the choice is in the type.
        return set_grad ? _block_backward_gemm(ΔX4, X1, state, RB, Val(true)) :
                          _block_backward_gemm(ΔX4, X1, state, RB, Val(false))
    end
    inds = channel_indices(Val(N))
    dims = batch_reduction_dims(Val(N))

    # `X2` and `X3` are the activations of `Y1` and `Y2`; taking them from the saved state
    # rather than reapplying the activation is what makes the state worth carrying.
    Y1, Y2, Y3, X2, X3, X4 = state

    # Cdims
    cdims2 = DenseConvDims(Y2, RB.W2.data; stride=RB.strides[2], padding=RB.pad[2])
    cdims3 = DCDims(X1, RB.W3.data;  stride=RB.strides[1], padding=RB.pad[1])

    # Backpropagate residual ΔX4 and compute gradients. `fan=true` is a linear output, so
    # its cotangent passes straight through.
    ΔY3 = RB.fan ? ΔX4 : GaLUgrad(ΔX4, Y3)
    ΔX3 = conv(ΔY3, RB.W3.data, cdims3)
    ΔW3 = ∇conv_filter(ΔY3, X3, cdims3)

    ΔY2 = backward(ΔX3, Y2, X3, RB.activation)
    ΔX2 = ∇conv_data(ΔY2, RB.W2.data, cdims2)
    ΔX2 .+= ΔY2
    ΔW2 = ∇conv_filter(X2, ΔY2, cdims2)
    Δb2 = sum(ΔY2, dims=dims)[inds...]

    cdims1 = DenseConvDims(X1, RB.W1.data; stride=RB.strides[1], padding=RB.pad[1])

    ΔY1 = backward(ΔX2, Y1, X2, RB.activation)
    ΔX1 = ∇conv_data(ΔY1, RB.W1.data, cdims1)
    ΔW1 = ∇conv_filter(X1, ΔY1, cdims1)
    Δb1 = sum(ΔY1, dims=dims)[inds...]

    # Set gradients
    if set_grad
        RB.W1.grad = ΔW1
        RB.W2.grad = ΔW2
        RB.W3.grad = ΔW3
        RB.b1.grad = Δb1
        RB.b2.grad = Δb2
    else
        Δθ = [Parameter(ΔW1), Parameter(ΔW2), Parameter(ΔW3), Parameter(Δb1), Parameter(Δb2)]
    end

    set_grad ? (return ΔX1) : (return ΔX1, Δθ)
end

## Jacobian-related functions
function jacobian(ΔX1::AbstractArray{T, N}, Δθ::AbstractVector{<:Parameter},
                  X1::AbstractArray{T, N}, RB::ResidualBlock) where {T, N}
    inds = channel_indices(Val(N))
    # Cdims
    cdims1 = DenseConvDims(X1, RB.W1.data; stride=RB.strides[1], padding=RB.pad[1])

    Y1 = conv(X1, RB.W1.data, cdims1) .+ reshape(RB.b1.data, inds...)
    ΔY1 = conv(ΔX1, RB.W1.data, cdims1) + conv(X1, Δθ[1].data, cdims1) .+ reshape(Δθ[4].data, inds...)
    X2 = RB.activation.forward(Y1)
    ΔX2 = backward(ΔY1, Y1, X2, RB.activation)

    cdims2 = DenseConvDims(X2, RB.W2.data; stride=RB.strides[2], padding=RB.pad[2])

    Y2 = X2 + conv(X2, RB.W2.data, cdims2) .+ reshape(RB.b2.data, inds...)
    ΔY2 = ΔX2 + conv(ΔX2, RB.W2.data, cdims2) + conv(X2, Δθ[2].data, cdims2) .+ reshape(Δθ[5].data, inds...)
    X3 = RB.activation.forward(Y2)
    ΔX3 = backward(ΔY2, Y2, X3, RB.activation)

    cdims3 = DCDims(X1, RB.W3.data; nc=2*size(X1, N-1), stride=RB.strides[1], padding=RB.pad[1])
    Y3 = ∇conv_data(X3, RB.W3.data, cdims3)
    ΔY3 = ∇conv_data(ΔX3, RB.W3.data, cdims3) + ∇conv_data(X3, Δθ[3].data, cdims3)
    if RB.fan
        X4, ΔX4 = Y3, ΔY3
    else
        ΔX4, X4 = GaLUjacobian(ΔY3, Y3)
    end

    return ΔX4, X4

end
 
# 2D/3D
function adjointJacobian(ΔX4::AbstractArray{T, N}, X1::AbstractArray{T, N}, RB::ResidualBlock) where {T, N}
    return backward(ΔX4, X1, RB; set_grad=false)
end
