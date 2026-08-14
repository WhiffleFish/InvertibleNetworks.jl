# Elementwise bijectors between a bounded interval and the real line.
#
# `SigmoidLayer` and friends in `utils/activation_functions.jl` are activations used *inside*
# a coupling block; they are not chain elements and carry no log-determinant. A flow over
# data with bounded support -- actions in [-1, 1], probabilities in [0, 1], anything squashed
# -- needs the squash to be part of the flow, so that its Jacobian is accounted for
# automatically instead of by hand outside the network.

export SigmoidBijector, TanhBijector

"""
    B = SigmoidBijector(; low=0f0, high=1f0, logdet=false)
    B = TanhBijector(; low=-1f0, high=1f0, logdet=false)

 Parameterless elementwise bijector between the interval `(low, high)` and the real line,
 usable as a layer of an [`InvertibleChain`](@ref).

 Following the convention of this package -- `forward` maps data to latent -- `forward` is
 the *unsquash* (logit or atanh) and `inverse` is the squash the layer is named for. So a
 flow over bounded data puts the bijector first:

     flow = InvertibleChain(SigmoidBijector(; low=-1f0, high=1f0, logdet=true),
                            ActNorm(n; logdet=true),
                            CouplingLayerGlow(n, n_hidden; logdet=true))

 and `flow.inverse(Z)` then generates samples that are inside `(low, high)` by construction,
 with `logdet` carrying the change of variables for free.

 *Input*:

 - `low`, `high`: bounds of the data interval

 - `logdet`: bool to indicate whether to compute the logdet

 *Usage:*

 - Forward mode: `Z, logdet = B.forward(X)` -- `X` must lie strictly inside `(low, high)`

 - Inverse mode: `X = B.inverse(Z)`

 - Backward mode: `ΔZ, Z = B.backward(ΔX, X)`

 *Trainable parameters:*

 - None

 See also: [`InvertibleChain`](@ref), [`log_likelihood`](@ref)
"""
struct BoundedBijector{K,T<:Real} <: NeuralNetLayer
    low::T
    high::T
    logdet::Bool
end

function BoundedBijector{K}(low::Real, high::Real, logdet::Bool) where {K}
    low < high || throw(ArgumentError("need low < high, got low = $low, high = $high"))
    l, h = promote(low, high)
    return BoundedBijector{K,typeof(l)}(l, h, logdet)
end

SigmoidBijector(; low=0f0, high=1f0, logdet=false) =
    BoundedBijector{:sigmoid}(low, high, logdet)
TanhBijector(; low=-1f0, high=1f0, logdet=false) =
    BoundedBijector{:tanh}(low, high, logdet)

Flux.@layer BoundedBijector trainable=()

Base.show(io::IO, B::BoundedBijector{K}) where {K} =
    print(io, "BoundedBijector{:$K}($(B.low), $(B.high))")

supports_per_sample_logdet(::BoundedBijector) = true

@inline _span(B::BoundedBijector, ::Type{T}) where {T} = T(B.high) - T(B.low)

# The interior coordinate each kind works in: `u ∈ (0,1)` for the sigmoid, `v ∈ (-1,1)` for
# the tanh. Clamped strictly inside, so a sample sitting exactly on a bound (or a rounding
# error just outside it) gives a large finite value rather than an Inf that poisons the
# whole batch.
@inline function _interior(X::AbstractArray{T,N}, B::BoundedBijector{:sigmoid}) where {T,N}
    e = eps(T)
    return clamp.((X .- T(B.low)) ./ _span(B, T), e, one(T) - e)
end

@inline function _interior(X::AbstractArray{T,N}, B::BoundedBijector{:tanh}) where {T,N}
    e = eps(T)
    return clamp.((T(2)*X .- T(B.low) .- T(B.high)) ./ _span(B, T), -one(T) + e, one(T) - e)
end

# log|dz/dx| elementwise, from the interior coordinate.
@inline _log_deriv(u, B::BoundedBijector{:sigmoid}, ::Type{T}) where {T} =
    .-log.(u) .- log1p.(.-u) .- log(_span(B, T))
@inline _log_deriv(v, B::BoundedBijector{:tanh}, ::Type{T}) where {T} =
    .-log1p.(.-v.^2) .+ log(T(2)/_span(B, T))

# dz/dx elementwise.
@inline _deriv(u, B::BoundedBijector{:sigmoid}, ::Type{T}) where {T} =
    one(T) ./ (_span(B, T) .* u .* (one(T) .- u))
@inline _deriv(v, B::BoundedBijector{:tanh}, ::Type{T}) where {T} =
    T(2) ./ (_span(B, T) .* (one(T) .- v.^2))

# d/dx log|dz/dx| elementwise: the log-determinant depends on where the sample is, so it
# contributes to the propagated residual even though the layer has no parameters.
@inline _dlog_deriv(u, B::BoundedBijector{:sigmoid}, ::Type{T}) where {T} =
    (T(2)*u .- one(T)) ./ (_span(B, T) .* u .* (one(T) .- u))
@inline _dlog_deriv(v, B::BoundedBijector{:tanh}, ::Type{T}) where {T} =
    T(4)*v ./ (_span(B, T) .* (one(T) .- v.^2))

# Interior coordinate recovered from the latent side, without going through `X`.
@inline _interior_from_latent(Z::AbstractArray{T,N}, ::BoundedBijector{:sigmoid}) where {T,N} =
    Sigmoid(Z)
@inline _interior_from_latent(Z::AbstractArray{T,N}, ::BoundedBijector{:tanh}) where {T,N} =
    tanh.(Z)

@inline _latent(u, ::BoundedBijector{:sigmoid}, ::Type{T}) where {T} = log.(u) .- log1p.(.-u)
@inline _latent(v, ::BoundedBijector{:tanh}, ::Type{T}) where {T} =
    (log1p.(v) .- log1p.(.-v))/T(2)

@inline _data(u, B::BoundedBijector{:sigmoid}, ::Type{T}) where {T} =
    T(B.low) .+ _span(B, T)*u
@inline _data(v, B::BoundedBijector{:tanh}, ::Type{T}) where {T} =
    T(B.low) .+ _span(B, T)*(v .+ one(T))/T(2)

# Forward pass: data -> latent
function forward(X::AbstractArray{T,N}, B::BoundedBijector; logdet=nothing) where {T,N}
    mode = logdet_mode(isnothing(logdet) ? B.logdet : logdet)
    p = _interior(X, B)
    Z = _latent(p, B, T)
    return _bijector_out(Z, p, B, mode, one(T))
end

# Inverse pass: latent -> data
function inverse(Z::AbstractArray{T,N}, B::BoundedBijector; logdet=false) where {T,N}
    mode = logdet_mode(logdet)
    p = _interior_from_latent(Z, B)
    X = _data(p, B, T)
    return _bijector_out(X, p, B, mode, -one(T))
end

_bijector_out(out, p, ::BoundedBijector, ::Val{false}, sign) = out
_bijector_out(out, p, B::BoundedBijector, ::Val{true}, sign::T) where {T} =
    (out, sign*sum(_log_deriv(p, B, T))/size(out, ndims(out)))
_bijector_out(out, p, B::BoundedBijector, ::Val{:sample}, sign::T) where {T} =
    (out, sign*per_sample_sum(_log_deriv(p, B, T)))

# Backward pass: Input (ΔZ, Z), Output (ΔX, X)
function backward(ΔZ::AbstractArray{T,N}, Z::AbstractArray{T,N}, B::BoundedBijector;
                  set_grad::Bool=true, logdet_weight=nothing) where {T,N}
    check_logdet_weight(logdet_weight, set_grad)
    p = _interior_from_latent(Z, B)
    X = _data(p, B, T)
    ΔX = ΔZ .* _deriv(p, B, T)
    B.logdet && (ΔX = ΔX .+ apply_logdet_weight(_dlog_deriv(p, B, T), logdet_weight))
    set_grad && return ΔX, X
    return ΔX, Parameter[], X
end

# Reverse-layer backward pass: Input (ΔX, X), Output (ΔZ, Z)
function backward_inv(ΔX::AbstractArray{T,N}, X::AbstractArray{T,N}, B::BoundedBijector;
                      set_grad::Bool=true, logdet_weight=nothing) where {T,N}
    check_logdet_weight(logdet_weight, set_grad)
    p = _interior(X, B)
    Z = _latent(p, B, T)
    deriv = _deriv(p, B, T)
    ΔZ = ΔX ./ deriv
    # The inverse pass carries `-logdet`, so the residual it adds is the forward one with
    # the opposite sign, pulled back through the map.
    B.logdet && (ΔZ = ΔZ .- apply_logdet_weight(_dlog_deriv(p, B, T), logdet_weight) ./ deriv)
    set_grad && return ΔZ, Z
    return ΔZ, Parameter[], Z
end
