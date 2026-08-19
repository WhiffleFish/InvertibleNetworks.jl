# Activation normalization layer
# Adapted from Kingma and Dhariwal (2018): https://arxiv.org/abs/1807.03039
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020
#

export ActNorm, reset!

"""
    AN = ActNorm(k; logdet=false)

 Create activation normalization layer. The parameters are initialized during
 the first use, such that the output has zero mean and unit variance along
 channels for the current mini-batch size.

 *Input*:

 - `k`: number of channels

 - `logdet`: bool to indicate whether to compute the logdet

 *Output*:

 - `AN`: Network layer for activation normalization.

 *Usage:*

 - Forward mode: `Y, logdet = AN.forward(X)`

 - Inverse mode: `X = AN.inverse(Y)`

 - Backward mode: `ΔX, X = AN.backward(ΔY, Y)`

 *Trainable parameters:*

 - Scaling factor `AN.s`

 - Bias `AN.b`

 See also: [`get_params`](@ref), [`clear_grad!`](@ref)
"""
mutable struct ActNorm{P<:Parameter} <: NeuralNetLayer
    k::Int64
    s::P
    b::P
    logdet::Bool
    is_reversed::Bool
    initialized::Bool
end

Flux.@layer ActNorm

needs_init(AN::ActNorm) = !AN.initialized
supports_per_sample_logdet(::ActNorm) = true

# `ActNorm` derives its parameters from the statistics of the first batch it sees in its
# normalizing direction (`forward`, or `inverse` once the layer is reversed). Until then the
# opposite direction has no map to apply: it would silently use s=1, b=0 and then change the
# map on the first normalizing pass, so say so instead.
function uninitialized_message(AN::ActNorm)
    normalizing, other = AN.is_reversed ? ("inverse", "forward") : ("forward", "inverse")
    return "ActNorm has not been initialized: its parameters are set from the statistics " *
           "of the first batch seen by `$normalizing`, so `$other` has no map to apply " *
           "yet. Run `init!(net, X)` on a representative batch first."
end

# Constructor: Initialize with nothing
function ActNorm(k; logdet=false)
    s = Parameter(ones(Float32, k))
    b = Parameter(zeros(Float32, k))
    return ActNorm(k, s, b, logdet, false, false)
end

function reset!(AN::ActNorm)
    fill!(AN.s.data, one(eltype(AN.s.data)))
    fill!(AN.b.data, zero(eltype(AN.b.data)))
    AN.s.grad = nothing
    AN.b.grad = nothing
    AN.initialized = false
    return AN
end

# 2-3D Foward pass: Input X, Output Y
function forward(X::AbstractArray{T, N}, AN::ActNorm; logdet=nothing) where {T, N}
    logdet = isnothing(logdet) ? (AN.logdet && ~AN.is_reversed) : logdet
    return _forward(X, AN, logdet_mode(logdet))
end

# The scaling is shared by the whole batch, so every sample gets the same log-determinant;
# `:sample` spreads it over the batch so it composes with the per-sample values of layers
# whose scaling does vary from sample to sample.
_actnorm_out(Y, ::AbstractArray, ::ActNorm, ::Val{false}) = Y
_actnorm_out(Y, X::AbstractArray{T, N}, AN::ActNorm, ::Val{true}) where {T, N} =
    (Y, logdet_forward(size(X)[1:N-2]..., AN.s))
_actnorm_out(Y, X::AbstractArray{T, N}, AN::ActNorm, ::Val{:sample}) where {T, N} =
    (Y, constant_per_sample(X, logdet_device(size(X)[1:N-2], AN.s)))

function _forward(X::AbstractArray{T, N}, AN::ActNorm, mode::Val) where {T, N}
    inds = channel_indices(Val(N))
    dims = batch_reduction_dims(Val(N))

    # Initialize during first pass such that
    # output has zero mean and unit variance
    if !AN.initialized
        # Mirror image of the check in `inverse`: a reversed layer normalizes on its inverse
        # pass, so for it this is the direction that is undefined before initialization.
        AN.is_reversed && throw(ArgumentError(uninitialized_message(AN)))
        μ = mean(X; dims=dims)[inds...]
        σ_sqr = var(X; dims=dims)[inds...]
        AN.s.data .= 1 ./ sqrt.(σ_sqr)
        AN.b.data .= -μ ./ sqrt.(σ_sqr)
        AN.initialized = true
    end
    Y = X .* reshape(AN.s.data, inds...) .+ reshape(AN.b.data, inds...)

    # If logdet true, return as second ouput argument
    return _actnorm_out(Y, X, AN, mode)
end

# 2-3D Inverse pass: Input Y, Output X
function inverse(Y::AbstractArray{T, N}, AN::ActNorm; logdet=nothing) where {T, N}
    isnothing(logdet) ? logdet = (AN.logdet && AN.is_reversed) : logdet = logdet
    mode = logdet_mode(logdet)
    inds = channel_indices(Val(N))
    dims = batch_reduction_dims(Val(N))

    # Initialize during first pass such that
    # output has zero mean and unit variance
    if !AN.initialized
        # Only a reversed layer normalizes on its inverse pass. For a forward-normalizing
        # layer the map is not defined until it has seen data in the forward direction:
        # running `inverse` first would silently use s=1, b=0 and then change the map on
        # the first forward pass.
        AN.is_reversed || throw(ArgumentError(uninitialized_message(AN)))
        μ = mean(Y; dims=dims)[inds...]
        σ_sqr = var(Y; dims=dims)[inds...]
        AN.s.data .= sqrt.(σ_sqr)
        AN.b.data .= μ
        AN.initialized = true
    end
    X = (Y .- reshape(AN.b.data, inds...)) ./ reshape(AN.s.data, inds...)

    # If logdet true, return as second ouput argument
    return _actnorm_inverse_out(X, Y, AN, mode)
end

_actnorm_inverse_out(X, ::AbstractArray, ::ActNorm, ::Val{false}) = X
_actnorm_inverse_out(X, Y::AbstractArray{T, N}, AN::ActNorm, ::Val{true}) where {T, N} =
    (X, -logdet_forward(size(Y)[1:N-2]..., AN.s))
_actnorm_inverse_out(X, Y::AbstractArray{T, N}, AN::ActNorm, ::Val{:sample}) where {T, N} =
    (X, constant_per_sample(Y, -logdet_device(size(Y)[1:N-2], AN.s)))

# 2-3D Backward pass: Input (ΔY, Y), Output (ΔY, Y)
function backward(ΔY::AbstractArray{T, N}, Y::AbstractArray{T, N}, AN::ActNorm; set_grad::Bool = true, logdet_weight=nothing) where {T, N}
    check_logdet_weight(logdet_weight, set_grad)
    inds = channel_indices(Val(N))
    dims = batch_reduction_dims(Val(N))
    nn = size(ΔY)[1:N-2]

    X = inverse(Y, AN; logdet=false)
    ΔX = ΔY .* reshape(AN.s.data, inds...)
    Δs = sum(ΔY .* X, dims=dims)[inds...]
    if AN.logdet
        set_grad ? (Δs += logdet_total_weight(logdet_weight)*logdet_backward(nn..., AN.s)) : (Δs_ = logdet_backward(nn..., AN.s))
    end
    Δb = sum(ΔY, dims=dims)[inds...]
    if set_grad
        AN.s.grad = Δs
        AN.b.grad = Δb
    else
        Δθ = [Parameter(Δs), Parameter(Δb)]
    end
    if set_grad
        return ΔX, X
    else
        AN.logdet ? (return ΔX, Δθ, X, [Parameter(Δs_), Parameter(0*Δb)]) : (return ΔX, Δθ, X)
    end
end

## Reverse-layer functions
# 2-3D Backward pass (inverse): Input (ΔX, X), Output (ΔX, X)
function backward_inv(ΔX::AbstractArray{T, N}, X::AbstractArray{T, N}, AN::ActNorm; set_grad::Bool = true, logdet_weight=nothing) where {T, N}
    check_logdet_weight(logdet_weight, set_grad)
    inds = channel_indices(Val(N))
    dims = batch_reduction_dims(Val(N))
    nn = size(ΔX)[1:N-2]

    Y = forward(X, AN; logdet=false)
    ΔY = ΔX ./ reshape(AN.s.data, inds...)
    Δs = -sum(ΔX .* X ./ reshape(AN.s.data, inds...), dims=dims)[inds...]
    if AN.logdet
        set_grad ? (Δs -= logdet_total_weight(logdet_weight)*logdet_backward(nn..., AN.s)) : (∇logdet = -logdet_backward(nn..., AN.s))
    end
    Δb = -sum(ΔX ./ reshape(AN.s.data, inds...), dims=dims)[inds...]
    if set_grad
        AN.s.grad = Δs
        AN.b.grad = Δb
    else
        Δθ = [Parameter(Δs), Parameter(Δb)]
    end
    if set_grad
        return ΔY, Y
    else
        AN.logdet ? (return ΔY, Δθ, Y, ∇logdet) : (return ΔY, Δθ, Y)
    end
end

## Jacobian-related functions
# 2-£D
function jacobian(ΔX::AbstractArray{T, N}, Δθ::AbstractVector{<:Parameter}, X::AbstractArray{T, N}, AN::ActNorm; logdet=nothing) where {T, N}
    isnothing(logdet) ? logdet = (AN.logdet && ~AN.is_reversed) : logdet = logdet
    inds = channel_indices(Val(N))
    nn = size(ΔX)[1:N-2]
    Δs = Δθ[1].data
    Δb = Δθ[2].data

    # Forward evaluation
    logdet ? (Y, lgdet) = forward(X, AN; logdet=logdet) : Y = forward(X, AN; logdet=logdet)

    # Jacobian evaluation
    ΔY = ΔX .* reshape(AN.s.data, inds...) .+ X .* reshape(Δs, inds...) .+ reshape(Δb, inds...)

    # Hessian evaluation of logdet terms
    if logdet
        nx, ny, _, _ = size(X)
        HlogΔθ = [Parameter(logdet_hessian(nn..., AN.s).*Δs), Parameter(zeros(Float32, size(Δb)))]
        return ΔY, Y, lgdet, HlogΔθ
    else
        return ΔY, Y
    end
end

# 2D/3D
function adjointJacobian(ΔY::AbstractArray{T, N}, Y::AbstractArray{T, N}, AN::ActNorm) where {T, N}
    return backward(ΔY, Y, AN; set_grad=false)
end


## Logdet utils
# 1D Logdet
logdet_forward(nx, s) = nx*sum(log.(abs.(s.data)))
logdet_backward(nx, s) = nx ./ s.data
logdet_hessian(nx, s) = -nx ./ s.data.^2f0
# 2D Logdet
logdet_forward(nx, ny, s) = nx*ny*sum(log.(abs.(s.data)))
logdet_backward(nx, ny, s) = nx*ny ./ s.data
logdet_hessian(nx, ny, s) = -nx*ny ./ s.data.^2f0
# 3D Logdet
logdet_forward(nx, ny, nz, s) = nx*ny*nz*sum(log.(abs.(s.data)))
logdet_backward(nx, ny, nz, s) = nx*ny*nz ./ s.data
logdet_hessian(nx, ny, nz, s) = -nx*ny*nz ./ s.data.^2f0

# Reverse
function tag_as_reversed!(AN::ActNorm, tag::Bool)
    AN.is_reversed = tag
    return AN
end
