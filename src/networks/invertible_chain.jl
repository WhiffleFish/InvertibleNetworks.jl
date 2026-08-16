# Chain-style composition of invertible layers, with automatic logdet accumulation
# and a ChainRules pullback so Flux/Zygote can differentiate it directly.

export InvertibleChain, flow_forward, flow_forward_per_sample, forward_per_sample
export supports_per_sample_logdet
export inverse_and_log_likelihood, inverse_and_log_likelihood_per_sample

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

 - Inverse mode: `X = C.inverse(Z)`, or `X, logdet = C.inverse(Z; logdet=true)`

 - Backward mode: `ΔX, X = C.backward(ΔZ, Z)`

 - Per sample: pass `logdet=:sample` to either direction for the vector of per-sample
   log-determinants instead of their batch average; see [`log_likelihood_per_sample`](@ref)
   and [`forward_per_sample`](@ref)

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

# A chain can be scored per sample when every layer that contributes a log-determinant can
# report it per sample; the rest have to be evaluated one sample at a time.
supports_per_sample_logdet(C::InvertibleChain) =
    all(l -> !contributes_logdet(l) || supports_per_sample_logdet(l), C.layers)

function _check_per_sample_logdet(C::InvertibleChain)
    supports_per_sample_logdet(C) && return nothing
    bad = first(l for l in C.layers if contributes_logdet(l) && !supports_per_sample_logdet(l))
    throw(ArgumentError(
        "$(nameof(typeof(bad))) reports its log-determinant averaged over the batch and " *
        "cannot report it per sample, so this chain has no per-sample log-determinant"))
end

# Recursion over the layer tuple: unrolled by the compiler, so no runtime tuple indexing.
@inline _forward_step(X, layer, ::Val{true}) = forward(X, layer)
@inline _forward_step(X, layer, mode::Val{:sample}) =
    contributes_logdet(layer) ? forward(X, layer; logdet=:sample) : forward(X, layer)

_apply_forward(X, logdet, ::Tuple{}, ::Val) = (X, logdet)
function _apply_forward(X, logdet, layers::Tuple, mode::Val)
    Y, Δlogdet = _step_out(_forward_step(X, first(layers), mode))
    return _apply_forward(Y, _accumulate_logdet(logdet, Δlogdet), Base.tail(layers), mode)
end

"""
    Z, logdet = forward(X, C::InvertibleChain)
    Z, logdet = forward(X, C::InvertibleChain; logdet=:sample)

 Apply the layers of `C` in order, accumulating the log-determinant of every layer built
 with `logdet=true`.

 By default `logdet` is the batch-averaged scalar the layers have always returned. With
 `logdet=:sample` it is instead the vector of length `size(X, N)` that scalar is the mean
 of, computed in the same single pass: the per-sample information is already in each
 layer's scaling, and only the reduction over the batch dimension destroys it.
"""
forward(X::AbstractArray{T,N}, C::InvertibleChain{L,LD}; logdet=nothing) where {T,N,L,LD} =
    _chain_forward(X, C, logdet_mode(logdet, Val(LD)))

_chain_forward(X::AbstractArray{T,N}, C::InvertibleChain, ::Val{false}) where {T,N} =
    _apply_forward(X, zero(T), C.layers, Val(true))[1]

_chain_forward(X::AbstractArray{T,N}, C::InvertibleChain, ::Val{true}) where {T,N} =
    _apply_forward(X, zero(T), C.layers, Val(true))

function _chain_forward(X::AbstractArray{T,N}, C::InvertibleChain, mode::Val{:sample}) where {T,N}
    _check_accumulates_logdet(C)
    _check_per_sample_logdet(C)
    return _apply_forward(X, constant_per_sample(X, zero(T)), C.layers, mode)
end

_check_accumulates_logdet(C::InvertibleChain) = C.logdet || throw(ArgumentError(
    "this network does not accumulate a log-determinant, so there is none to report; " *
    "build its layers with logdet=true"))

_apply_inverse(Z, ::Tuple{}) = Z
function _apply_inverse(Z, layers::Tuple)
    X, _ = _step_out(inverse(Z, first(layers)))
    return _apply_inverse(X, Base.tail(layers))
end

# Log-determinant of the inverse map, for the layers that can report it. Every layer that
# contributes to the forward log-determinant recomputes its scaling on the inverse pass
# anyway, so this is free; the generic fallback throws rather than silently dropping a term.
_inverse_with_logdet(Z::AbstractArray, L::Union{ActNorm,Conv1x1,CouplingLayerGlow,AffineLayer,
                                                BoundedBijector,CouplingLayerSpline,SplineLayer}, mode) =
    inverse(Z, L; logdet=_mode_kwarg(mode))
_inverse_with_logdet(Z::AbstractArray, C::InvertibleChain, mode) =
    inverse(Z, C; logdet=_mode_kwarg(mode))
_inverse_with_logdet(::AbstractArray, L, ::Any) = throw(ArgumentError(
    "$(nameof(typeof(L))) contributes a log-determinant but cannot report it on the " *
    "inverse pass; get it from a forward pass instead of asking `inverse` for it"))

@inline _mode_kwarg(::Val{true}) = true
@inline _mode_kwarg(::Val{:sample}) = :sample

# A layer contributes the same term to both directions, with opposite signs, so the
# inverse total is the negative of the forward total for the same chain.
@inline function _inverse_step(Z, layer, mode::Val)
    contributes_logdet(layer) || return (_step_out(inverse(Z, layer))[1], nothing)
    return _inverse_with_logdet(Z, layer, mode)
end

_apply_inverse(X, logdet, ::Tuple{}, ::Val) = (X, logdet)
function _apply_inverse(Z, logdet, layers::Tuple, mode::Val)
    X, Δlogdet = _inverse_step(Z, first(layers), mode)
    return _apply_inverse(X, _accumulate_logdet(logdet, Δlogdet), Base.tail(layers), mode)
end

"""
    X = inverse(Z, C::InvertibleChain)
    X, logdet = inverse(Z, C::InvertibleChain; logdet=true)
    X, logdet = inverse(Z, C::InvertibleChain; logdet=:sample)

 Apply the layers of `C` in reverse, each in its inverse direction.

 With `logdet=true` the log-determinant of the inverse map is returned alongside `X`, so a
 sample and its density come out of a single pass. It is the negative of what a forward
 pass over the same chain reports, hence the minus sign in the change of variables:

     log p_X(X) = log p_Z(Z) - logdet

 `logdet=:sample` returns the per-sample vector instead of the batch-averaged scalar.

 See also: [`inverse_and_log_likelihood`](@ref), [`log_likelihood`](@ref)
"""
inverse(Z::AbstractArray{T,N}, C::InvertibleChain; logdet=false) where {T,N} =
    _inverse(Z, C, logdet_mode(logdet))

_inverse(Z::AbstractArray{T,N}, C::InvertibleChain, ::Val{false}) where {T,N} =
    _apply_inverse(Z, reverse(C.layers))

function _inverse(Z::AbstractArray{T,N}, C::InvertibleChain, ::Val{true}) where {T,N}
    _check_inverse_logdet(C)
    return _apply_inverse(Z, zero(T), reverse(C.layers), Val(true))
end

function _inverse(Z::AbstractArray{T,N}, C::InvertibleChain, mode::Val{:sample}) where {T,N}
    _check_inverse_logdet(C)
    _check_per_sample_logdet(C)
    return _apply_inverse(Z, constant_per_sample(Z, zero(T)), reverse(C.layers), mode)
end

_check_inverse_logdet(C::InvertibleChain) = _check_accumulates_logdet(C)

# Per-sample log-determinant weights, when given, are handed to each layer that contributes
# one: the hand-written backward passes cannot recover them from a batch-averaged pass.
@inline _backward_step(ΔY, Y, layer, ::Nothing) = backward(ΔY, Y, layer)
@inline _backward_step(ΔY, Y, layer, w::AbstractVector) =
    contributes_logdet(layer) ? backward(ΔY, Y, layer; logdet_weight=w) : backward(ΔY, Y, layer)

_apply_backward(ΔY, Y, ::Tuple{}, w) = (ΔY, Y)
function _apply_backward(ΔY, Y, layers::Tuple, w)
    ΔX, X = _backward_step(ΔY, Y, first(layers), w)
    return _apply_backward(ΔX, X, Base.tail(layers), w)
end

function backward(ΔZ::AbstractArray{T,N}, Z::AbstractArray{T,N}, C::InvertibleChain;
                  set_grad::Bool=true, logdet_weight=nothing) where {T,N}
    set_grad || throw(ArgumentError("InvertibleChain only implements backward with " *
                                    "set_grad=true; use the layers directly for the " *
                                    "Jacobian interface"))
    return _apply_backward(ΔZ, Z, reverse(C.layers), logdet_weight)
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

# A loss may use only the log-determinant, leaving no cotangent on `Z` at all.
_input_cotangent(ΔZ, Z::AbstractArray) = convert(typeof(Z), ΔZ)
_input_cotangent(::AbstractZero, Z::AbstractArray) = zero(Z)
_input_cotangent(::Nothing, Z::AbstractArray) = zero(Z)

function ChainRulesCore.rrule(::typeof(flow_forward), net::Invertible, X, θ)
    out = forward(X, net)
    pullback(Δout) = _flow_pullback(net, out, unthunk(Δout))
    return out, pullback
end


"""
    Z, logdet = flow_forward_per_sample(net, X, θ)

 Differentiable entry point for a network's per-sample log-determinant: the same single
 forward pass as [`flow_forward`](@ref), with `logdet` a vector of length `size(X, N)`
 instead of its batch average.

 Its pullback takes the per-sample cotangent seriously. The batch-averaged path can rescale
 a log-determinant gradient after the fact, because every sample carries the same weight;
 per-example weights cannot be recovered that way, so the weights are pushed into the
 layers' backward passes instead.
"""
flow_forward_per_sample(net::Invertible, X::AbstractArray, ::Any) =
    forward(X, net; logdet=:sample)


"""
    Z, logdet = forward_per_sample(X, net)

 Forward pass reporting a per-sample log-determinant: a vector of length `size(X, N)` rather
 than its batch average. Differentiable with Zygote/Flux, including with per-example weights
 on `logdet`.

 `net.forward(X; logdet=:sample)` computes the same thing but, like every hand-written
 `forward` in this package, cannot be traced by AD; this is the entry point to use inside a
 loss.

# Example

```julia
function loss(m)
    Z, logdet = forward_per_sample(X, m)
    return -sum(weights .* (log_likelihood_per_sample(Z) .+ logdet))
end
```

 which is [`log_likelihood_per_sample`](@ref)`(X, m)` spelled out.

 See also: [`log_likelihood_per_sample`](@ref), [`flow_forward`](@ref)
"""
forward_per_sample(X::AbstractArray, net::Invertible) =
    flow_forward_per_sample(net, X, parameter_data(net))

function ChainRulesCore.rrule(::typeof(flow_forward_per_sample), net::Invertible, X, θ)
    out = forward(X, net; logdet=:sample)
    function flow_forward_per_sample_pullback(Δout)
        ΔZ, Δlogdet = _output_cotangent(unthunk(Δout))
        Z = out[1]
        ΔX, Δθ = _flow_backward_weighted(net, _input_cotangent(ΔZ, Z), Z,
                                         _per_sample_weight(Δlogdet, out[2]))
        return NoTangent(), NoTangent(), ΔX, Δθ
    end
    return out, flow_forward_per_sample_pullback
end

_per_sample_weight(w::AbstractVector, ::AbstractVector) = w
_per_sample_weight(::AbstractZero, logdet::AbstractVector) = zero(logdet)
_per_sample_weight(::Nothing, logdet::AbstractVector) = zero(logdet)

# One pass: with explicit per-sample weights the layers compute the gradient the loss
# actually asked for, so there is nothing to correct afterwards.
function _flow_backward_weighted(net::Invertible, ΔZ, Z, w)
    params = get_params(net)
    # `Conv1x1` accumulates into `p.grad` where every other layer overwrites. Cleared through
    # `params` rather than through `net`, which would walk the network a second time.
    clear_grad!(params)
    ΔX, _ = backward(copy(ΔZ), copy(Z), net; logdet_weight=w)
    return ΔX, getfield.(params, :grad)
end

# Network without a log-determinant: the cotangent is just ΔZ.
function _flow_pullback(net::Invertible, Z::AbstractArray, Δout)
    ΔX, Δθ = _flow_backward(net, _input_cotangent(Δout, Z), Z, -1)
    return NoTangent(), NoTangent(), ΔX, Δθ
end

# Network returning (Z, logdet): honor whatever weight the loss gave the log-determinant.
function _flow_pullback(net::Invertible, out::Tuple, Δout)
    length(out) == 2 || throw(ArgumentError(
        "flow_forward expects a network whose forward returns Z or (Z, logdet); got a " *
        "$(length(out))-tuple. Conditional networks are not supported."))
    Z = out[1]
    ΔZ, Δlogdet = _output_cotangent(Δout)
    ΔX, Δθ = _flow_backward(net, _input_cotangent(ΔZ, Z), Z, _logdet_weight(Δlogdet))
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
    clear_grad!(params)
    ΔX, _ = backward(copy(ΔZ), copy(Z), net)
    Δθ = getfield.(params, :grad)

    if weight != -1
        clear_grad!(params)
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
    X, f = inverse_and_log_likelihood(Z, net; μ=0f0, σ=1f0, normalized=false)

 Push the latent sample `Z` through `net.inverse` and return both the sample `X` and its
 log-likelihood under the flow, from a single pass:

     log p_X(X) = log p_Z(Z) - logdet

 where `logdet` is the log-determinant of the inverse map. This is what a sampling loop
 wants -- generating an `X` and scoring it are the same computation, and doing them
 separately costs a second pass through the network.

 The value follows the conventions of [`log_likelihood`](@ref): averaged over the batch,
 and up to an additive constant unless `normalized=true`.

# Example

```julia
Z = randn(Float32, nx, ny, n, batchsize)
X, logp = inverse_and_log_likelihood(Z, flow)
```

 See also: [`log_likelihood`](@ref), [`InvertibleChain`](@ref)
"""
function inverse_and_log_likelihood(Z::AbstractArray{T,N}, net::Invertible; μ=T(0), σ=T(1),
                                    normalized::Bool=false) where {T,N}
    X, logdet = _inverse_with_logdet_checked(Z, net, true)
    return X, log_likelihood(Z; μ=μ, σ=σ, normalized=normalized) - logdet
end


"""
    X, f = inverse_and_log_likelihood_per_sample(Z, net; μ=0f0, σ=1f0, normalized=false)

 Per-sample form of [`inverse_and_log_likelihood`](@ref): `f` is a vector of length
 `size(Z, N)` holding the log-likelihood of each generated sample.

 This is what a rollout wants -- each generated sample scored individually, from the pass
 that generated it.

 See also: [`inverse_and_log_likelihood`](@ref), [`log_likelihood_per_sample`](@ref)
"""
function inverse_and_log_likelihood_per_sample(Z::AbstractArray{T,N}, net::Invertible;
                                               μ=T(0), σ=T(1),
                                               normalized::Bool=false) where {T,N}
    X, logdet = _inverse_with_logdet_checked(Z, net, :sample)
    return X, log_likelihood_per_sample(Z; μ=μ, σ=σ, normalized=normalized) .- logdet
end

function _inverse_with_logdet_checked(Z::AbstractArray, net::Invertible, mode)
    out = inverse(Z, net; logdet=mode)
    out isa Tuple || throw(ArgumentError(
        "inverse_and_log_likelihood needs the change-of-variables term, but this network " *
        "does not accumulate a log-determinant; build its layers with logdet=true"))
    return out
end


"""
    f = log_likelihood_per_sample(X, net; μ=0f0, σ=1f0, normalized=false)

 Log-likelihood of each sample of `X` under the normalizing flow `net`, returned as a vector
 of length `size(X, N)`. This is the unaggregated form of [`log_likelihood`](@ref)`(X, net)`:

     sum(log_likelihood_per_sample(X, net))/size(X, N) == log_likelihood(X, net)

 Useful for per-example scoring -- outlier/anomaly detection, importance weights, or
 inspecting which samples the flow explains badly -- and differentiable, so per-example
 weights can appear in a loss.

 This is a single batched pass whenever every log-determinant in `net` can be reported per
 sample (see [`supports_per_sample_logdet`](@ref)). For a network whose layers only report
 a batch average, it falls back to evaluating `net` one sample at a time: exact, since the
 layers do not mix samples, but N times the work.

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
    init!(net, X)
    return _log_likelihood_per_sample(X, net, Val(supports_per_sample_logdet(net)); μ=μ, σ=σ,
                                      normalized=normalized)
end

function _log_likelihood_per_sample(X::AbstractArray{T,N}, net::Invertible, ::Val{true};
                                    μ=T(0), σ=T(1), normalized::Bool=false) where {T,N}
    Z, logdet = flow_forward_per_sample(net, X, parameter_data(net))
    return log_likelihood_per_sample(Z; μ=μ, σ=σ, normalized=normalized) .+ logdet
end

# One pass per sample, for networks whose layers only report a batch-averaged
# log-determinant.
function _log_likelihood_per_sample(X::AbstractArray{T,N}, net::Invertible, ::Val{false};
                                    μ=T(0), σ=T(1), normalized::Bool=false) where {T,N}
    colons = ntuple(_ -> Colon(), Val(N-1))
    return [log_likelihood(X[colons..., i:i], net; μ=μ, σ=σ, normalized=normalized)
            for i in axes(X, N)]
end
