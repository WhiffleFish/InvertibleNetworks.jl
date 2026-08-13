# Chain-style composition of invertible layers, with automatic logdet accumulation
# and a ChainRules pullback so Flux/Zygote can differentiate it directly.

export InvertibleChain, flow_forward

"""
    C = InvertibleChain(layers...)

 Compose invertible layers/networks into a single invertible network, applying them in the
 order given (like `Flux.Chain`, and unlike `∘`/[`Composition`](@ref), which composes
 right-to-left). The log-determinant of every layer that was built with `logdet=true` is
 accumulated automatically.

 *Input*:

 - `layers`: invertible layers or networks, in the order they should be applied

 *Output*:

 - `C`: invertible network

 *Usage:*

 - Forward mode: `Z, logdet = C.forward(X)`, or just `Z = C.forward(X)` when no layer
   contributes a log-determinant

 - Inverse mode: `X = C.inverse(Z)`

 - Backward mode: `ΔX, X = C.backward(ΔZ, Z)`

 Unlike the other networks in this package, `C(X)` returns the same thing whether or not it
 is called inside a gradient, so a flow objective can be written once and used for both
 training and evaluation:

```julia
flow = InvertibleChain(ActNorm(n; logdet=true), CouplingLayerGlow(n, n_hidden; logdet=true))

function loss(flow, X)
    Z, logdet = flow(X)
    return -log_likelihood(Z) - logdet
end

opt_state = Flux.setup(Adam(1f-3), flow)
for i = 1:niter
    l, grads = Flux.withgradient(m -> loss(m, X), flow)
    Flux.update!(opt_state, flow, grads[1])
end
```

 *Trainable parameters:*

 - The parameters of each layer in `C.layers`

 See also: [`Composition`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct InvertibleChain{L<:Tuple,LD} <: InvertibleNetwork
    layers::L
    logdet::Bool
end

# `logdet` is carried as a type parameter so that `forward` infers whether it returns
# `Z` or `(Z, logdet)`; it is also kept as a field so the generic `Invertible` machinery
# (and Functors reconstruction) sees a plain struct.
InvertibleChain(layers::L, logdet::Bool) where {L<:Tuple} =
    InvertibleChain{L,logdet}(layers, logdet)

InvertibleChain(layers::Tuple) = InvertibleChain(layers, any(contributes_logdet, layers))
InvertibleChain(layers::Invertible...) = InvertibleChain(layers)

Flux.@layer InvertibleChain

# Whether a layer adds a log-determinant to a forward pass. A layer tagged as reversed
# reports its log-determinant on the inverse pass instead, matching each layer's own
# `logdet && ~is_reversed` convention.
function contributes_logdet(layer)
    hasproperty(layer, :logdet) || return false
    layer.logdet || return false
    return !(hasproperty(layer, :is_reversed) && layer.is_reversed)
end


## Chain-like interface

length(C::InvertibleChain) = length(C.layers)
getindex(C::InvertibleChain, i::Integer) = C.layers[i]
getindex(C::InvertibleChain, i) = InvertibleChain(C.layers[i])
Base.firstindex(::InvertibleChain) = 1
Base.lastindex(C::InvertibleChain) = length(C.layers)
Base.iterate(C::InvertibleChain, state...) = iterate(C.layers, state...)
Base.eltype(::Type{<:InvertibleChain{L}}) where {L} = eltype(L)

function Base.show(io::IO, C::InvertibleChain)
    print(io, "InvertibleChain(")
    for (i, layer) in enumerate(C.layers)
        i > 1 && print(io, ", ")
        print(io, typeof(layer).name.name)
    end
    print(io, ")")
end


## Forward/inverse/backward

# Each layer returns either `Y` or `(Y, logdet)` depending on how it was built, so
# normalize to a `(Y, Δlogdet)` pair with `nothing` standing for "no contribution".
@inline _step_out(out::AbstractArray) = (out, nothing)
@inline _step_out(out::Tuple) = (out[1], out[2])
@inline _accumulate_logdet(logdet, ::Nothing) = logdet
@inline _accumulate_logdet(logdet, Δlogdet) = logdet + Δlogdet

# Recursion over the layer tuple: unrolled by the compiler, so no runtime tuple indexing.
_apply_forward(X, logdet, ::Tuple{}) = (X, logdet)
function _apply_forward(X, logdet, layers::Tuple)
    Y, Δlogdet = _step_out(forward(X, first(layers)))
    return _apply_forward(Y, _accumulate_logdet(logdet, Δlogdet), Base.tail(layers))
end

function forward(X::AbstractArray{T,N}, C::InvertibleChain{L,LD}) where {T,N,L,LD}
    Z, logdet = _apply_forward(X, zero(T), C.layers)
    return LD ? (Z, logdet) : Z
end

_apply_inverse(Z, ::Tuple{}) = Z
function _apply_inverse(Z, layers::Tuple)
    X, _ = _step_out(inverse(Z, first(layers)))
    return _apply_inverse(X, Base.tail(layers))
end

inverse(Z::AbstractArray{T,N}, C::InvertibleChain) where {T,N} =
    _apply_inverse(Z, reverse(C.layers))

_apply_backward(ΔY, Y, ::Tuple{}) = (ΔY, Y)
function _apply_backward(ΔY, Y, layers::Tuple)
    ΔX, X = backward(ΔY, Y, first(layers))
    return _apply_backward(ΔX, X, Base.tail(layers))
end

function backward(ΔZ::AbstractArray{T,N}, Z::AbstractArray{T,N}, C::InvertibleChain;
                  set_grad::Bool=true) where {T,N}
    set_grad || throw(ArgumentError("InvertibleChain only implements backward with " *
                                    "set_grad=true; use the layers directly for the " *
                                    "Jacobian interface"))
    return _apply_backward(ΔZ, Z, reverse(C.layers))
end

function jacobian(::AbstractArray{T,N}, ::AbstractVector{<:Parameter}, ::AbstractArray{T,N},
                  ::InvertibleChain) where {T,N}
    throw(ArgumentError("Jacobian for InvertibleChain not yet implemented; use " *
                        "Composition for the Jacobian interface"))
end


## Automatic differentiation
#
# The hand-written `backward` of each layer computes the gradient of an objective that
# contains `-logdet` with unit weight, no matter what the user's loss actually says. That
# is fine for the usual flow objective, but it silently disagrees with, say, a loss that
# ignores the log-determinant. `backward` is affine in its cotangent -- `backward(ΔZ)` =
# `A(ΔZ) - B` with `A` linear and `B` the log-determinant gradient -- so a second pass with
# a zero cotangent recovers `B` and lets any log-determinant weight be honored exactly.

"""
    out = flow_forward(net, X, θ)

 Differentiable entry point for an unconditional invertible network: returns exactly what
 `net.forward(X)` returns -- `(Z, logdet)` for a network that accumulates a
 log-determinant, `Z` otherwise -- both inside and outside a gradient.

 This differs from calling `net(X)` on the other network types, which under AD returns `Z`
 alone and routes the log-determinant through [`logdetjac`](@ref) and the global tape.
"""
flow_forward(net::Invertible, X::AbstractArray, ::Any) = forward(X, net)

(C::InvertibleChain)(X::AbstractArray) = flow_forward(C, X, parameter_data(C))

# Cotangent of a `(Z, logdet)` output. Zygote hands over a `Tangent`, but accept a plain
# tuple too so the pullback can be exercised directly.
_output_cotangent(Δ::Tangent) = (unthunk(Δ[1]), unthunk(Δ[2]))
_output_cotangent(Δ::Tuple) = (unthunk(Δ[1]), unthunk(Δ[2]))

_logdet_weight(::AbstractZero) = 0
_logdet_weight(::Nothing) = 0
_logdet_weight(w::Number) = w

function ChainRulesCore.rrule(::typeof(flow_forward), net::Invertible, X, θ)
    out = forward(X, net)
    pullback(Δout) = _flow_pullback(net, out, unthunk(Δout))
    return out, pullback
end

# Network without a log-determinant: the cotangent is just ΔZ.
function _flow_pullback(net::Invertible, Z::AbstractArray, Δout)
    ΔX, Δθ = _flow_backward(net, convert(typeof(Z), Δout), Z, -1)
    return NoTangent(), NoTangent(), ΔX, Δθ
end

# Network returning (Z, logdet): honor whatever weight the loss gave the log-determinant.
function _flow_pullback(net::Invertible, out::Tuple, Δout)
    length(out) == 2 || throw(ArgumentError(
        "flow_forward expects a network whose forward returns Z or (Z, logdet); got a " *
        "$(length(out))-tuple. Conditional networks are not supported."))
    Z = out[1]
    ΔZ, Δlogdet = _output_cotangent(Δout)
    ΔX, Δθ = _flow_backward(net, convert(typeof(Z), ΔZ), Z, _logdet_weight(Δlogdet))
    return NoTangent(), NoTangent(), ΔX, Δθ
end

# The hand-written `backward` of each layer computes the gradient of an objective that
# contains `-logdet` with unit weight, whatever the user's loss actually says. `backward` is
# affine in its cotangent -- `backward(ΔZ) = A(ΔZ) - B` with `A` linear and `B` the
# log-determinant gradient -- so a second pass with a zero cotangent recovers `B` and lets
# any log-determinant weight be honored exactly.
function _flow_backward(net::Invertible, ΔZ, Z, weight)
    params = get_params(net)
    # `Conv1x1` accumulates into `p.grad` where every other layer overwrites, so the slate
    # has to be clean for each pass: otherwise leftover gradients (or the first pass's own
    # results) leak into the numbers below.
    clear_grad!(net)
    ΔX, _ = backward(copy(ΔZ), copy(Z), net)
    Δθ = getfield.(params, :grad)

    if weight != -1
        clear_grad!(net)
        ΔX_offset, _ = backward(zero(ΔZ), copy(Z), net)
        Δθ_offset = getfield.(params, :grad)
        correction = 1 + weight
        ΔX = ΔX .- correction .* ΔX_offset
        Δθ = _correct_grads(Δθ, Δθ_offset, correction)
        # Leave the layers holding the gradient that matches the user's loss.
        for (p, g) in zip(params, Δθ)
            p.grad = g
        end
    end

    return ΔX, Δθ
end

_correct_grads(Δθ, Δθ_offset, correction) =
    map((g, g0) -> isnothing(g) || isnothing(g0) ? g : g .- correction .* g0, Δθ, Δθ_offset)


## Log-likelihood of the data under a flow

"""
    f = log_likelihood(X, net; μ=0f0, σ=1f0, normalized=false)

 Log-likelihood of the data `X` under the normalizing flow `net`, obtained by change of
 variables: for `Z, logdet = net.forward(X)`,

     log p_X(X) = log p_Z(Z) + log|det J|

 so this returns `log_likelihood(Z; μ=μ, σ=σ) + logdet`. Use `-log_likelihood(X, net)` as
 the training objective; it is differentiable with Zygote/Flux, and the gradient accounts
 for the log-determinant term.

 `net` must accumulate a log-determinant, i.e. its layers must be built with `logdet=true`.

 Follows the same conventions as the latent-space [`log_likelihood`](@ref): the value is
 averaged over the batch, and by default the Gaussian normalizing constant is dropped, so
 the result is a log-density up to an additive constant. Pass `normalized=true` for a
 calibrated log-density -- with it, `exp` of this value integrates to 1 over the data space.
 The constant does not depend on the parameters, so it does not affect gradients.

# Example

```julia
flow = InvertibleChain(ActNorm(n; logdet=true), CouplingLayerGlow(n, n_hidden; logdet=true))
flow(X)                                      # initialize ActNorm

opt_state = Flux.setup(Adam(1f-3), flow)
l, grads = Flux.withgradient(m -> -log_likelihood(X, m), flow)
Flux.update!(opt_state, flow, grads[1])
```

 See also: [`log_likelihood`](@ref), [`InvertibleChain`](@ref)
"""
function log_likelihood(X::AbstractArray{T,N}, net::Invertible; μ=T(0), σ=T(1),
                        normalized::Bool=false) where {T,N}
    out = flow_forward(net, X, parameter_data(net))
    return _flow_log_likelihood(out, μ, σ, normalized)
end

_flow_log_likelihood(out::Tuple, μ, σ, normalized) =
    log_likelihood(out[1]; μ=μ, σ=σ, normalized=normalized) + out[2]

_flow_log_likelihood(::AbstractArray, μ, σ, normalized) = throw(ArgumentError(
    "log_likelihood(X, net) needs the change-of-variables term, but this network does not " *
    "accumulate a log-determinant; build its layers with logdet=true"))


"""
    f = log_likelihood_per_sample(X, net; μ=0f0, σ=1f0, normalized=false)

 Log-likelihood of each sample of `X` under the normalizing flow `net`, returned as a vector
 of length `size(X, N)`. This is the unaggregated form of [`log_likelihood`](@ref)`(X, net)`:

     sum(log_likelihood_per_sample(X, net))/size(X, N) == log_likelihood(X, net)

 Useful for per-example scoring -- outlier/anomaly detection, importance weights, or
 inspecting which samples the flow explains badly.

 The layers report a batch-aggregated log-determinant (`coupling_logdet_forward` divides by
 the batch), so there is no per-sample log-determinant to read off a single batched pass.
 These layers do not mix samples, though, so `net` is evaluated one sample at a time, which
 is exact. The total work is the same as one batched pass -- only the per-call overhead is
 multiplied, measured at roughly 1.3x for a batch of 32.

 A full-batch forward pass is run first so that lazily-initialized layers (`ActNorm`) take
 their statistics from the whole batch rather than from a single sample.

# Example

```julia
scores = log_likelihood_per_sample(X, flow; normalized=true)
outliers = findall(<(quantile(scores, 0.05)), scores)
```

 See also: [`log_likelihood`](@ref), [`log_likelihood_per_sample`](@ref)
"""
function log_likelihood_per_sample(X::AbstractArray{T,N}, net::Invertible; μ=T(0), σ=T(1),
                                  normalized::Bool=false) where {T,N}
    # Initialize lazily-initialized layers from the whole batch. Hidden from AD: it is a
    # setup side effect, and tracing the hand-written forward directly (rather than through
    # the `flow_forward` rule) would hit its in-place updates.
    ChainRulesCore.ignore_derivatives() do
        forward(X, net)
    end
    colons = ntuple(_ -> Colon(), Val(N-1))
    return [log_likelihood(X[colons..., i:i], net; μ=μ, σ=σ, normalized=normalized)
            for i in axes(X, N)]
end
