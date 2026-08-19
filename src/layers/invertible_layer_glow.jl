# Affine coupling layer from Dinh et al. (2017)
# Includes 1x1 convolution from in Putzky and Welling (2019)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export CouplingLayerGlow, CouplingLayerGlow3D


"""
    CL = CouplingLayerGlow(C::Conv1x1, RB::ResidualBlock; logdet=false)

or

    CL = CouplingLayerGlow(n_in, n_hidden; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, logdet=false, ndims=2) (2D)

    CL = CouplingLayerGlow(n_in, n_hidden; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, logdet=false, ndims=3) (3D)
    
    CL = CouplingLayerGlow3D(n_in, n_hidden; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, logdet=false) (3D)

 Create a Real NVP-style invertible coupling layer based on 1x1 convolutions and a residual block.

 *Input*:

 - `C::Conv1x1`: 1x1 convolution layer

 - `RB::ResidualBlock`: residual block layer consisting of 3 convolutional layers with ReLU activations.

 - `logdet`: bool to indicate whether to compte the logdet of the layer

 or

 - `n_in`, `n_hidden`: number of input and hidden channels

 - `k1`, `k2`: kernel size of convolutions in residual block. `k1` is the kernel of the first and third
    operator, `k2` is the kernel size of the second operator.

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second convolution (`p2`)

 - `s1`, `s2`: stride for the first and third convolution (`s1`) and the second convolution (`s2`)

 - `ndims` : number of dimensions

 *Output*:

 - `CL`: Invertible Real NVP coupling layer.

 *Usage:*

 - Forward mode: `Y, logdet = CL.forward(X)`    (if constructed with `logdet=true`)

 - Inverse mode: `X = CL.inverse(Y)`

 - Backward mode: `ΔX, X = CL.backward(ΔY, Y)`

 *Trainable parameters:*

 - None in `CL` itself

 - Trainable parameters in residual block `CL.RB` and 1x1 convolution layer `CL.C`

 See also: [`Conv1x1`](@ref), [`ResidualBlock`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct CouplingLayerGlow{C<:Conv1x1,R<:ConditionerBlock,A<:ActivationFunction,LD} <: NeuralNetLayer
    C::C
    RB::R
    logdet::Bool
    activation::A
end

CouplingLayerGlow(C::CT, RB::RT, logdet::Bool, activation::AT) where
    {CT<:Conv1x1,RT<:ConditionerBlock,AT<:ActivationFunction} =
    CouplingLayerGlow{CT,RT,AT,logdet}(C, RB, logdet, activation)

Flux.@layer CouplingLayerGlow

supports_per_sample_logdet(::CouplingLayerGlow) = true

# Constructor from a 1x1 convolution and a conditioner block
function CouplingLayerGlow(C::Conv1x1, RB::ConditionerBlock; logdet=false,
                           activation::ActivationFunction=SigmoidLayer())
    _check_fan(RB)
    return CouplingLayerGlow(C, RB, logdet, activation)
end

# Constructor from input dimensions
function CouplingLayerGlow(n_in::Int64, n_hidden::Int64; nx=nothing, dense=false, freeze_conv=false, k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, logdet=false, activation::ActivationFunction=SigmoidLayer(), ndims=2)

    # A coupling layer splits its channels in two and conditions one half on the other,
    # which needs at least two channels: `n_in = 1` gives an empty split and fails later,
    # inside the residual block, with an unrelated-looking error.
    n_in < 2 && throw(ArgumentError(
        "CouplingLayerGlow needs at least 2 input channels to split, got n_in = $n_in"))

    # 1x1 Convolution and residual block for invertible layer
    C = Conv1x1(n_in; freeze=freeze_conv)

    split_num = Int(round(n_in/2))
    in_chan   = n_in-split_num
    out_chan  = 2*split_num

    if dense
        # `nx` is the spatial extent the dense block flattens over. A flow on plain
        # `(dim, batch)` vectors has none, so it defaults to 1 rather than being required.
        d = isnothing(nx) ? 1 : prod(nx)
        RB = MLPBlock(d*in_chan, n_hidden; d_out=d*out_chan, fan=true)
    else 
        RB = ResidualBlock(in_chan, n_hidden;n_out=out_chan, k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, fan=true, ndims=ndims)
    end

    return CouplingLayerGlow(C, RB, logdet, activation)
end

CouplingLayerGlow3D(args...;kw...) = CouplingLayerGlow(args...; kw..., ndims=3)

# Forward pass: Input X, Output Y
function forward(X::AbstractArray{T, N}, L::CouplingLayerGlow{C,R,A,LD}; logdet=nothing) where {T,N,C,R,A,LD}
    return _forward(X, L, logdet_mode(logdet, Val(LD)))
end

# `Sm` varies from sample to sample, so unlike `ActNorm` this layer has a genuinely
# per-sample log-determinant; the scalar it returns by default is its batch average.
_glow_out(Y, ::AbstractArray, ::Val{false}) = Y
_glow_out(Y, Sm::AbstractArray, ::Val{true}) = (Y, glow_logdet_forward(Sm))
_glow_out(Y, Sm::AbstractArray, ::Val{:sample}) = (Y, logdet_per_sample(Sm))

# Only `X2` needs to be a real array: it feeds the conditioner's convolutions (or matrix
# multiplies), which need contiguous storage. `X1` and the conditioner's two output halves are
# only ever broadcast over, so they are views, and the transformed half is written straight
# into the output rather than built separately and then concatenated. Between them that is
# roughly one and a half copies of the whole tensor, per layer, that no longer happen.
function _forward(X::AbstractArray{T, N}, L::CouplingLayerGlow, mode::Val) where {T,N}
    X_ = conv1x1_forward(X, L.C)
    k = channel_split_index(X_)
    X1 = channel_view(X_, 1:k)                    # broadcast over only
    X2 = channel_copy(X_, channel_tail(X_, k))    # feeds the conditioner

    logSm, Tm = channel_halves(block_forward(X2, L.RB))
    Sm = L.activation.forward(logSm)

    Y = similar(X_)
    channel_view(Y, 1:k) .= Sm .* X1 .+ Tm
    copyto!(channel_view(Y, channel_tail(Y, k)), X2)

    return _glow_out(Y, Sm, mode)
end

# Inverse pass: Input Y, Output X
#
# `Sm` is recomputed here anyway, so the log-determinant of the inverse map costs nothing
# beyond the reduction; pass `logdet=true` to get it. It defaults to `false` rather than to
# `L.logdet` so that the `save=true` call in `backward` keeps its return shape.
function inverse(Y::AbstractArray{T, N}, L::CouplingLayerGlow; save=false, logdet=false) where {T,N}
    save && (return _inverse(Y, L, Val(true))[1:5])
    X, _, _, _, Sm, _ = _inverse(Y, L, Val(false))
    return _glow_inverse_out(X, Sm, logdet_mode(logdet))
end

# Shared inverse, with `save` selecting how much of the recomputed state comes back.
# `backward` asks for the saved residual-block state (`Val(true)`) so that the block's
# forward pass, which happens here regardless, is not run a second time inside its backward.
function _inverse(Y::AbstractArray{T, N}, L::CouplingLayerGlow, ::Val{save}) where {T,N,save}
    k = channel_split_index(Y)
    Y1 = channel_view(Y, 1:k)                    # broadcast over only
    X2 = channel_copy(Y, channel_tail(Y, k))     # the identity branch

    rb = save ? block_forward_save(X2, L.RB) : block_forward(X2, L.RB)
    logSm, Tm = channel_halves(block_output(rb))
    Sm = L.activation.forward(logSm)

    # `conv1x1_inverse` contracts the channel axis with a GEMM, so its input has to be a real
    # array; assembling it directly is one allocation instead of a half-sized one plus a
    # concatenation.
    X_ = similar(Y)
    X1 = channel_view(X_, 1:k)
    X1 .= (Y1 .- Tm) ./ (Sm .+ eps(T))   # epsilon to avoid division by 0
    copyto!(channel_view(X_, channel_tail(X_, k)), X2)

    X = conv1x1_inverse(X_, L.C)

    return X, X1, X2, logSm, Sm, rb
end

_glow_inverse_out(X, ::AbstractArray, ::Val{false}) = X
_glow_inverse_out(X, Sm::AbstractArray, ::Val{true}) = (X, -glow_logdet_forward(Sm))
_glow_inverse_out(X, Sm::AbstractArray, ::Val{:sample}) = (X, -logdet_per_sample(Sm))

# Backward pass: Input (ΔY, Y), Output (ΔX, X)
function backward(ΔY::AbstractArray{T, N}, Y::AbstractArray{T, N}, L::CouplingLayerGlow; set_grad::Bool=true, logdet_weight=nothing) where {T,N}
    check_logdet_weight(logdet_weight, set_grad)

    # Recompute forward state, keeping the residual block's own intermediates: its backward
    # pass below would otherwise run its forward a second time.
    X, X1, X2, logS, S, rb = _inverse(Y, L, Val(true))

    d = channel_dim(ΔY)
    k = channel_split_index(ΔY)
    ΔY1, ΔY2 = channel_halves(ΔY)

    # Backpropagate residual. `ΔT` is `ΔY1` itself: the cotangent handed to the conditioner is
    # assembled into fresh storage below, so there is nothing to copy here.
    ΔT = ΔY1
    ΔS = ΔY1 .* X1
    if L.logdet && set_grad
        ΔS .+= logdet_scale_grad(S, logdet_weight)
    end

    ΔX1 = ΔY1 .* S
    ΔlogS = backward(ΔS, logS, S, L.activation)

    # The conditioner's cotangent, `(ΔlogS, ΔT)` concatenated along the channel axis.
    ΔlogS_T = similar(ΔlogS, ntuple(i -> i == d ? 2k : size(ΔlogS, i), Val(N))...)
    copyto!(channel_view(ΔlogS_T, 1:k), ΔlogS)
    copyto!(channel_view(ΔlogS_T, k+1:2k), ΔT)

    if set_grad
        ΔX2 = block_backward(ΔlogS_T, X2, rb, L.RB)
        ΔX2 .+= ΔY2
    else
        ΔX2, Δθrb = block_backward(ΔlogS_T, X2, rb, L.RB; set_grad=false)
        ΔX2 .+= ΔY2
    end

    # `d(logdet)/dθ` is a property of the parameters alone, so its cotangent is the
    # log-determinant's own derivative with respect to the scaling -- not `ΔlogS`, which
    # carries the data cotangent and made this term depend on `ΔY`. Two passes are needed
    # because the two cotangents are not proportional; they share `rb`, so the conditioner's
    # forward pass is still paid for once.
    if L.logdet && !set_grad
        ΔlogS_ld = backward(glow_logdet_backward(S), logS, S, L.activation)
        copyto!(channel_view(ΔlogS_T, 1:k), ΔlogS_ld)   # reuses the buffer from above
        fill!(channel_view(ΔlogS_T, k+1:2k), zero(T))
        ∇logdet = block_backward(ΔlogS_T, X2, rb, L.RB; set_grad=false)[2]
    end

    ΔX_ = tensor_cat(ΔX1, ΔX2)
    if set_grad
        # `X` is already the 1x1 convolution's inverse of the concatenated halves, from above.
        ΔX = inverse_grad(ΔX_, X, L.C)
        return ΔX, X
    end

    ΔX, Δθc = inverse_grad(ΔX_, X, L.C; set_grad=false)
    Δθ = cat(Δθc, Δθrb; dims=1)
    L.logdet ? (return ΔX, Δθ, X, cat(0*Δθ[1:3], ∇logdet; dims=1)) : (return ΔX, Δθ, X)
end


## Jacobian-related functions

function jacobian(ΔX::AbstractArray{T, N}, Δθ::AbstractVector{<:Parameter}, X, L::CouplingLayerGlow) where {T,N}

    # Get dimensions
    k = Int(L.C.k/2)

    ΔX_, X_ = L.C.jacobian(ΔX, Δθ[1:3], X)
    X1, X2 = tensor_split(X_)
    ΔX1, ΔX2 = tensor_split(ΔX_)

    Y2 = copy(X2)
    ΔY2 = copy(ΔX2)
    ΔlogS_T, logS_T = L.RB.jacobian(ΔX2, Δθ[4:end], X2)
    ΔlogS, ΔlogT = tensor_split(ΔlogS_T)
    logS, logT = tensor_split(logS_T)
    Sm = L.activation.forward(logS)
    ΔS = backward(ΔlogS, logS, Sm, L.activation)
    Tm = logT
    ΔT = ΔlogT
    Y1 = Sm.*X1 + Tm
    ΔY1 = ΔS.*X1 + Sm.*ΔX1 + ΔT
    Y = tensor_cat(Y1, Y2)
    ΔY = tensor_cat(ΔY1, ΔY2)

    # Gauss-Newton approximation of logdet terms
    JΔθ,_ = tensor_split(L.RB.jacobian(cuzeros(ΔX2, size(ΔX2)), Δθ[4:end], X2)[1])#[:, :, 1:k, :]
    GNΔθ = cat(0f0*Δθ[1:3], -L.RB.adjointJacobian(tensor_cat(backward(JΔθ, logS, Sm, L.activation), zeros(Float32, size(Sm))), X2)[2]; dims=1)

    L.logdet ? (return ΔY, Y, glow_logdet_forward(Sm), GNΔθ) : (return ΔY, Y)
end

function adjointJacobian(ΔY::AbstractArray{T, N}, Y::AbstractArray{T, N}, L::CouplingLayerGlow) where {T, N}
    return backward(ΔY, Y, L; set_grad=false)
end

# Logdet (correct?)
glow_logdet_forward(S) = sum(log.(abs.(S))) / size(S)[end]
glow_logdet_backward(S) = 1f0./ S / size(S)[end]
