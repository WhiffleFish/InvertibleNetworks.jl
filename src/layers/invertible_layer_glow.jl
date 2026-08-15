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
struct CouplingLayerGlow{C<:Conv1x1,R<:Union{ResidualBlock,FluxBlock},A<:ActivationFunction,LD} <: NeuralNetLayer
    C::C
    RB::R
    logdet::Bool
    activation::A
end

CouplingLayerGlow(C::CT, RB::RT, logdet::Bool, activation::AT) where
    {CT<:Conv1x1,RT<:Union{ResidualBlock,FluxBlock},AT<:ActivationFunction} =
    CouplingLayerGlow{CT,RT,AT,logdet}(C, RB, logdet, activation)

Flux.@layer CouplingLayerGlow

supports_per_sample_logdet(::CouplingLayerGlow) = true

# Constructor from 1x1 convolution and residual block
function CouplingLayerGlow(C::Conv1x1, RB::ResidualBlock; logdet=false, activation::ActivationFunction=SigmoidLayer())
    RB.fan == false && throw("Set ResidualBlock.fan == true")
    return CouplingLayerGlow(C, RB, logdet, activation)
end

# Constructor from 1x1 convolution and residual Flux block
CouplingLayerGlow(C::Conv1x1, RB::FluxBlock; logdet=false, activation::ActivationFunction=SigmoidLayer()) = CouplingLayerGlow(C, RB, logdet, activation)

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
        isnothing(nx) && error("Dense network needs nx as kwarg input")
        RB = FluxBlock(Chain(x->reshape(x,nx*in_chan,:),Dense(nx*in_chan,n_in*n_hidden,relu),Dense(n_in*n_hidden,n_in*n_hidden,relu),Dense(n_in*n_hidden,nx*out_chan,relu),x->reshape(x,nx,out_chan,:)))
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

function _forward(X::AbstractArray{T, N}, L::CouplingLayerGlow, mode::Val) where {T,N}
    X_ = conv1x1_forward(X, L.C)
    X1, X2 = tensor_split(X_)

    logS_T = block_forward(X2, L.RB)
    logSm, Tm = tensor_split(logS_T)
    Sm = L.activation.forward(logSm)
    Y1 = @. Sm * X1 + Tm

    # The second half passes through unchanged, and `tensor_cat` copies it anyway.
    Y = tensor_cat(Y1, X2)

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
    Y1, X2 = tensor_split(Y)   # the second half is the identity branch

    rb = save ? block_forward_save(X2, L.RB) : block_forward(X2, L.RB)
    logSm, Tm = tensor_split(block_output(rb))
    Sm = L.activation.forward(logSm)
    X1 = @. (Y1 - Tm) / (Sm + eps(T)) # add epsilon to avoid division by 0

    X = conv1x1_inverse(tensor_cat(X1, X2), L.C)

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

    # Backpropagate residual
    ΔY1, ΔY2 = tensor_split(ΔY)
    ΔT = ΔY1                    # `tensor_cat` below copies it, so no copy is needed here
    ΔS = ΔY1 .* X1
    if L.logdet
        set_grad ? (ΔS .+= logdet_scale_grad(S, logdet_weight)) : (ΔS_ = glow_logdet_backward(S))
    end

    ΔX1 = ΔY1 .* S
    ΔlogS = backward(ΔS, logS, S, L.activation)
    if set_grad
        ΔX2 = block_backward(tensor_cat(ΔlogS, ΔT), X2, rb, L.RB)
        ΔX2 .+= ΔY2
    else
        # Both passes reuse `rb`, so the block's forward pass is paid for once, not twice.
        ΔX2, Δθrb = block_backward(tensor_cat(ΔlogS, ΔT), X2, rb, L.RB; set_grad=set_grad)
        _, ∇logdet = block_backward(tensor_cat(ΔlogS, 0f0.*ΔT), X2, rb, L.RB; set_grad=set_grad)
        ΔX2 += ΔY2
    end
    ΔX_ = tensor_cat(ΔX1, ΔX2)
    if set_grad
        # `X` is already the 1x1 convolution's inverse of `tensor_cat(X1, X2)`, from above.
        ΔX = inverse_grad(ΔX_, X, L.C)
    else
        ΔX, Δθc = inverse_grad(ΔX_, X, L.C; set_grad=set_grad)
        Δθ = cat(Δθc, Δθrb; dims=1)
    end

    if set_grad
        return ΔX, X
    else
        L.logdet ? (return ΔX, Δθ, X, cat(0*Δθ[1:3], ∇logdet; dims=1)) : (return ΔX, Δθ, X)
    end
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
