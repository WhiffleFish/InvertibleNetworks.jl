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

# Conditioning
#
# One context tensor is shared by every layer: `nothing` is the unconditional chain, and it is
# what the two-argument entry points pass, so both share one set of recursions. The two levels
# of dispatch below are both on TYPES -- `Nothing` against `AbstractArray`, then the layer's own
# type -- so the unconditional path compiles to what it compiled to before, and a conditional
# chain has no runtime branch either.
#
# A layer with no conditioner ignores the context. That is what lets a conditional chain hold an
# `ActNorm` or a `SplineLayer` unchanged, which matters because those are the layers a spline
# stack is normally interleaved with.
@inline _consumes_context(::Any) = false
@inline _consumes_context(L::CouplingLayerSpline) = L.n_ctx > 0

@inline _ctx_forward(X, ::Nothing, layer, mode) = _forward_step(X, layer, mode)
@inline _ctx_forward(X, Ctx::AbstractArray, layer, mode) = _forward_step(X, layer, mode)
@inline _ctx_forward(X, Ctx::AbstractArray, layer::CouplingLayerSpline, ::Val{true}) =
    forward(X, Ctx, layer)
@inline _ctx_forward(X, Ctx::AbstractArray, layer::CouplingLayerSpline, ::Val{:sample}) =
    contributes_logdet(layer) ? forward(X, Ctx, layer; logdet=:sample) : forward(X, Ctx, layer)

# Recursion over the layer tuple: unrolled by the compiler, so no runtime tuple indexing.
@inline _forward_step(X, layer, ::Val{true}) = forward(X, layer)
@inline _forward_step(X, layer, mode::Val{:sample}) =
    contributes_logdet(layer) ? forward(X, layer; logdet=:sample) : forward(X, layer)

_apply_forward(X, Ctx, logdet, ::Tuple{}, ::Val) = (X, logdet)
function _apply_forward(X, Ctx, logdet, layers::Tuple, mode::Val)
    Y, Δlogdet = _step_out(_ctx_forward(X, Ctx, first(layers), mode))
    return _apply_forward(Y, Ctx, _accumulate_logdet(logdet, Δlogdet), Base.tail(layers), mode)
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
    _chain_forward(X, nothing, C, logdet_mode(logdet, Val(LD)))

forward(X::AbstractArray{T,N}, Ctx::AbstractArray{T,N}, C::InvertibleChain{L,LD};
        logdet=nothing) where {T,N,L,LD} =
    _chain_forward(X, Ctx, C, logdet_mode(logdet, Val(LD)))

_chain_forward(X::AbstractArray{T,N}, Ctx, C::InvertibleChain, ::Val{false}) where {T,N} =
    _apply_forward(X, Ctx, zero(T), C.layers, Val(true))[1]

_chain_forward(X::AbstractArray{T,N}, Ctx, C::InvertibleChain, ::Val{true}) where {T,N} =
    _apply_forward(X, Ctx, zero(T), C.layers, Val(true))

function _chain_forward(X::AbstractArray{T,N}, Ctx, C::InvertibleChain,
                        mode::Val{:sample}) where {T,N}
    _check_accumulates_logdet(C)
    _check_per_sample_logdet(C)
    return _apply_forward(X, Ctx, constant_per_sample(X, zero(T)), C.layers, mode)
end

_check_accumulates_logdet(C::InvertibleChain) = C.logdet || throw(ArgumentError(
    "this network does not accumulate a log-determinant, so there is none to report; " *
    "build its layers with logdet=true"))

@inline _ctx_inverse(Z, ::Nothing, layer) = inverse(Z, layer)
@inline _ctx_inverse(Z, Ctx::AbstractArray, layer) = inverse(Z, layer)
@inline _ctx_inverse(Z, Ctx::AbstractArray, layer::CouplingLayerSpline) = inverse(Z, Ctx, layer)

_apply_inverse(Z, Ctx, ::Tuple{}) = Z
function _apply_inverse(Z, Ctx, layers::Tuple)
    X, _ = _step_out(_ctx_inverse(Z, Ctx, first(layers)))
    return _apply_inverse(X, Ctx, Base.tail(layers))
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
@inline function _inverse_step(Z, Ctx, layer, mode::Val)
    contributes_logdet(layer) || return (_step_out(_ctx_inverse(Z, Ctx, layer))[1], nothing)
    return _ctx_inverse_with_logdet(Z, Ctx, layer, mode)
end

@inline _ctx_inverse_with_logdet(Z, ::Nothing, layer, mode) = _inverse_with_logdet(Z, layer, mode)
@inline _ctx_inverse_with_logdet(Z, Ctx::AbstractArray, layer, mode) =
    _inverse_with_logdet(Z, layer, mode)
@inline _ctx_inverse_with_logdet(Z, Ctx::AbstractArray, L::CouplingLayerSpline, mode) =
    inverse(Z, Ctx, L; logdet=_mode_kwarg(mode))

_apply_inverse(X, Ctx, logdet, ::Tuple{}, ::Val) = (X, logdet)
function _apply_inverse(Z, Ctx, logdet, layers::Tuple, mode::Val)
    X, Δlogdet = _inverse_step(Z, Ctx, first(layers), mode)
    return _apply_inverse(X, Ctx, _accumulate_logdet(logdet, Δlogdet), Base.tail(layers), mode)
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
    _inverse(Z, nothing, C, logdet_mode(logdet))

inverse(Z::AbstractArray{T,N}, Ctx::AbstractArray{T,N}, C::InvertibleChain;
        logdet=false) where {T,N} =
    _inverse(Z, Ctx, C, logdet_mode(logdet))

_inverse(Z::AbstractArray{T,N}, Ctx, C::InvertibleChain, ::Val{false}) where {T,N} =
    _apply_inverse(Z, Ctx, reverse(C.layers))

function _inverse(Z::AbstractArray{T,N}, Ctx, C::InvertibleChain, ::Val{true}) where {T,N}
    _check_inverse_logdet(C)
    return _apply_inverse(Z, Ctx, zero(T), reverse(C.layers), Val(true))
end

function _inverse(Z::AbstractArray{T,N}, Ctx, C::InvertibleChain, mode::Val{:sample}) where {T,N}
    _check_inverse_logdet(C)
    _check_per_sample_logdet(C)
    return _apply_inverse(Z, Ctx, constant_per_sample(Z, zero(T)), reverse(C.layers), mode)
end

_check_inverse_logdet(C::InvertibleChain) = _check_accumulates_logdet(C)

# Per-sample log-determinant weights, when given, are handed to each layer that contributes
# one: the hand-written backward passes cannot recover them from a batch-averaged pass.
@inline _backward_step(ΔY, Y, layer, ::Nothing) = backward(ΔY, Y, layer)
@inline _backward_step(ΔY, Y, layer, w::AbstractVector) =
    contributes_logdet(layer) ? backward(ΔY, Y, layer; logdet_weight=w) : backward(ΔY, Y, layer)

# One context feeds every conditional layer, so its cotangents ADD. Accumulating rather than
# overwriting is the difference between a correct gradient and one that only looks correct at
# depth 1 -- and a one-layer test would not catch it.
@inline _accumulate_ctx(::Nothing, ::Nothing) = nothing
@inline _accumulate_ctx(ΔCtx, ::Nothing) = ΔCtx
@inline _accumulate_ctx(::Nothing, Δc) = Δc
@inline _accumulate_ctx(ΔCtx, Δc) = ΔCtx .+ Δc

@inline function _ctx_backward(ΔY, Y, ::Nothing, layer, w)
    ΔX, X = _backward_step(ΔY, Y, layer, w)
    return ΔX, nothing, X
end
@inline function _ctx_backward(ΔY, Y, Ctx::AbstractArray, layer, w)
    ΔX, X = _backward_step(ΔY, Y, layer, w)
    return ΔX, nothing, X
end
@inline _ctx_backward(ΔY, Y, Ctx::AbstractArray, layer::CouplingLayerSpline, ::Nothing) =
    backward(ΔY, Y, Ctx, layer)
@inline _ctx_backward(ΔY, Y, Ctx::AbstractArray, layer::CouplingLayerSpline, w::AbstractVector) =
    contributes_logdet(layer) ? backward(ΔY, Y, Ctx, layer; logdet_weight=w) :
                                backward(ΔY, Y, Ctx, layer)

_apply_backward(ΔY, Y, Ctx, ΔCtx, ::Tuple{}, w) = (ΔY, ΔCtx, Y)
function _apply_backward(ΔY, Y, Ctx, ΔCtx, layers::Tuple, w)
    ΔX, Δc, X = _ctx_backward(ΔY, Y, Ctx, first(layers), w)
    return _apply_backward(ΔX, X, Ctx, _accumulate_ctx(ΔCtx, Δc), Base.tail(layers), w)
end

function backward(ΔZ::AbstractArray{T,N}, Z::AbstractArray{T,N}, C::InvertibleChain;
                  set_grad::Bool=true, logdet_weight=nothing) where {T,N}
    _check_backward_set_grad(set_grad)
    ΔX, _, X = _apply_backward(ΔZ, Z, nothing, nothing, reverse(C.layers), logdet_weight)
    return ΔX, X
end

"""
    ΔX, ΔCtx, X = backward(ΔZ, Z, Ctx, C::InvertibleChain; logdet_weight=nothing)

 Conditional backward pass: as the unconditional form, with the cotangent of the shared context
 alongside. `ΔCtx` is the SUM over the layers that read it, and is zero when none does.
"""
function backward(ΔZ::AbstractArray{T,N}, Z::AbstractArray{T,N}, Ctx::AbstractArray{T,N},
                  C::InvertibleChain; set_grad::Bool=true, logdet_weight=nothing) where {T,N}
    _check_backward_set_grad(set_grad)
    ΔX, ΔCtx, X = _apply_backward(ΔZ, Z, Ctx, nothing, reverse(C.layers), logdet_weight)
    return ΔX, _ctx_cotangent(ΔCtx, Ctx), X
end

_check_backward_set_grad(set_grad::Bool) =
    set_grad || throw(ArgumentError("InvertibleChain only implements backward with " *
                                    "set_grad=true; use the layers directly for the " *
                                    "Jacobian interface"))

# A chain in which nothing reads the context still owes its caller a tangent of the right shape.
@inline _ctx_cotangent(::Nothing, Ctx) = zero(Ctx)
@inline _ctx_cotangent(ΔCtx, ::Any) = ΔCtx

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

"""
    Z, logdet = flow_forward_per_sample(net, X, Ctx, θ)

 Conditional form of [`flow_forward_per_sample`](@ref): the same single forward pass, with a
 context tensor threaded into the layers that read one.

 Its pullback returns a cotangent for `Ctx` as well as for `X` and the parameters, which is
 what lets a trainable state encoder sit in front of the flow and be differentiated by ordinary
 Zygote from there. Without it the encoder would be frozen SILENTLY -- the flow's own
 parameters would still move and the loss would still fall.

 Only the per-sample form is conditional. The batch-averaged [`flow_forward`](@ref) recovers an
 arbitrary log-determinant weight with a second, zero-cotangent pass and a correction, which
 would have to be extended to the context as well; the per-sample path pushes explicit weights
 into the layers instead and needs no correction, so it is the one worth having conditional.
"""
flow_forward_per_sample(net::Invertible, X::AbstractArray, Ctx::AbstractArray, ::Any) =
    forward(X, Ctx, net; logdet=:sample)

function ChainRulesCore.rrule(::typeof(flow_forward_per_sample), net::Invertible, X, Ctx, θ)
    out = forward(X, Ctx, net; logdet=:sample)
    function flow_forward_per_sample_cond_pullback(Δout)
        ΔZ, Δlogdet = _output_cotangent(unthunk(Δout))
        Z = out[1]
        ΔX, ΔCtx, Δθ = _flow_backward_weighted(net, _input_cotangent(ΔZ, Z), Z, Ctx,
                                               _per_sample_weight(Δlogdet, out[2]))
        return NoTangent(), NoTangent(), ΔX, ΔCtx, Δθ
    end
    return out, flow_forward_per_sample_cond_pullback
end

function _flow_backward_weighted(net::Invertible, ΔZ, Z, Ctx::AbstractArray, w)
    params = get_params(net)
    clear_grad!(params)
    ΔX, ΔCtx, _ = backward(copy(ΔZ), copy(Z), Ctx, net; logdet_weight=w)
    return ΔX, ΔCtx, getfield.(params, :grad)
end

"""
    Z, logdet = forward_per_sample(X, Ctx, net)

 Conditional form of [`forward_per_sample`](@ref). Differentiable with Zygote/Flux in the
 parameters of `net` and in `Ctx`.
"""
forward_per_sample(X::AbstractArray, Ctx::AbstractArray, net::Invertible) =
    flow_forward_per_sample(net, X, Ctx, parameter_data(net))

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
    f = log_likelihood(X, net; μ=0f0, σ=1f0, normalized=false, base=nothing)

 Log-likelihood of the data `X` under the normalizing flow `net`, obtained by change of
 variables: for `Z, logdet = net.forward(X)`,

     log p_X(X) = log p_Z(Z) + log|det J|

 so this returns `log_likelihood(Z; base=base) + logdet`. Use `-log_likelihood(X, net)` as
 the training objective; it is differentiable with Zygote/Flux, and the gradient accounts
 for the log-determinant term.

 `net` must accumulate a log-determinant, i.e. its layers must be built with `logdet=true`.

 Follows the same conventions as the latent-space [`log_likelihood`](@ref): the value is
 averaged over the batch, and by default the normalizing constant is dropped, so the result
 is a log-density up to an additive constant. Pass `normalized=true` for a calibrated
 log-density -- with it, `exp` of this value integrates to 1 over the data space. The
 constant does not depend on the parameters, so it does not affect gradients.

 `base` selects the latent distribution and defaults to `StandardNormal(μ, σ)`. A base with
 bounded support additionally requires `net` to be a bijection of that support, which is
 checked here rather than left to produce quietly wrong numbers; see
 [`check_latent_support`](@ref).

# Example

```julia
flow = InvertibleChain(ActNorm(n; logdet=true), CouplingLayerGlow(n, n_hidden; logdet=true))
flow(X)                                      # initialize ActNorm

opt_state = Flux.setup(Adam(1f-3), flow)
l, grads = Flux.withgradient(m -> -log_likelihood(X, m), flow)
Flux.update!(opt_state, flow, grads[1])
```

 See also: [`log_likelihood`](@ref), [`LatentDistribution`](@ref), [`InvertibleChain`](@ref)
"""
function log_likelihood(X::AbstractArray{T,N}, net::Invertible; μ=T(0), σ=T(1),
                        normalized::Bool=false, base=nothing) where {T,N}
    d = _latent_base(base, μ, σ, T)
    check_latent_support(net, d)
    out = flow_forward(net, X, parameter_data(net))
    return _flow_log_likelihood(out, d, normalized)
end

_flow_log_likelihood(out::Tuple, d, normalized) =
    logpdf_mean(d, out[1]; normalized=normalized) + out[2]

_flow_log_likelihood(::AbstractArray, d, normalized) = throw(ArgumentError(
    "log_likelihood(X, net) needs the change-of-variables term, but this network does not " *
    "accumulate a log-determinant; build its layers with logdet=true"))


"""
    X, f = inverse_and_log_likelihood(Z, net; μ=0f0, σ=1f0, normalized=false, base=nothing)

 Push the latent sample `Z` through `net.inverse` and return both the sample `X` and its
 log-likelihood under the flow, from a single pass:

     log p_X(X) = log p_Z(Z) - logdet

 where `logdet` is the log-determinant of the inverse map. This is what a sampling loop
 wants -- generating an `X` and scoring it are the same computation, and doing them
 separately costs a second pass through the network.

 The value follows the conventions of [`log_likelihood`](@ref): averaged over the batch,
 and up to an additive constant unless `normalized=true`. `base` selects the latent
 distribution, defaulting to `StandardNormal(μ, σ)`; draw `Z` from the same one with
 `rand(base, dims...)`.

# Example

```julia
base = StandardNormal()
Z = rand(base, nx, ny, n, batchsize)
X, logp = inverse_and_log_likelihood(Z, flow; base=base)
```

 See also: [`log_likelihood`](@ref), [`LatentDistribution`](@ref), [`InvertibleChain`](@ref)
"""
function inverse_and_log_likelihood(Z::AbstractArray{T,N}, net::Invertible; μ=T(0), σ=T(1),
                                    normalized::Bool=false, base=nothing) where {T,N}
    d = _latent_base(base, μ, σ, T)
    check_latent_support(net, d)
    X, logdet = _inverse_with_logdet_checked(Z, net, true)
    return X, logpdf_mean(d, Z; normalized=normalized) - logdet
end


"""
    X, f = inverse_and_log_likelihood_per_sample(Z, net; μ=0f0, σ=1f0, normalized=false,
                                                 base=nothing)

 Per-sample form of [`inverse_and_log_likelihood`](@ref): `f` is a vector of length
 `size(Z, N)` holding the log-likelihood of each generated sample.

 This is what a rollout wants -- each generated sample scored individually, from the pass
 that generated it.

 See also: [`inverse_and_log_likelihood`](@ref), [`log_likelihood_per_sample`](@ref)
"""
function inverse_and_log_likelihood_per_sample(Z::AbstractArray{T,N}, net::Invertible;
                                               μ=T(0), σ=T(1), normalized::Bool=false,
                                               base=nothing) where {T,N}
    d = _latent_base(base, μ, σ, T)
    check_latent_support(net, d)
    X, logdet = _inverse_with_logdet_checked(Z, net, :sample)
    return X, logpdf_per_sample(d, Z; normalized=normalized) .- logdet
end

function _inverse_with_logdet_checked(Z::AbstractArray, net::Invertible, mode)
    out = inverse(Z, net; logdet=mode)
    out isa Tuple || throw(ArgumentError(
        "inverse_and_log_likelihood needs the change-of-variables term, but this network " *
        "does not accumulate a log-determinant; build its layers with logdet=true"))
    return out
end


"""
    f = log_likelihood_per_sample(X, net; μ=0f0, σ=1f0, normalized=false, base=nothing)

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

 `base` selects the latent distribution and defaults to `StandardNormal(μ, σ)`, on the same
 terms as [`log_likelihood`](@ref)`(X, net)` -- including the support check a bounded base
 brings with it.

# Example

```julia
scores = log_likelihood_per_sample(X, flow; normalized=true)
outliers = findall(<(quantile(scores, 0.05)), scores)
```

 See also: [`log_likelihood`](@ref), [`log_likelihood_per_sample`](@ref)
"""
function log_likelihood_per_sample(X::AbstractArray{T,N}, net::Invertible; μ=T(0), σ=T(1),
                                  normalized::Bool=false, base=nothing) where {T,N}
    d = _latent_base(base, μ, σ, T)
    check_latent_support(net, d)
    # Initialize lazily-initialized layers from the whole batch. Hidden from AD: it is a
    # setup side effect, and tracing the hand-written forward directly (rather than through
    # the `flow_forward` rule) would hit its in-place updates.
    init!(net, X)
    return _log_likelihood_per_sample(X, net, Val(supports_per_sample_logdet(net)), d,
                                      normalized)
end

function _log_likelihood_per_sample(X::AbstractArray{T,N}, net::Invertible, ::Val{true},
                                    d, normalized::Bool) where {T,N}
    Z, logdet = flow_forward_per_sample(net, X, parameter_data(net))
    return logpdf_per_sample(d, Z; normalized=normalized) .+ logdet
end

"""
    f = log_likelihood_per_sample(X, Ctx, net; μ=0f0, σ=1f0, normalized=false, base=nothing)

 Conditional form of [`log_likelihood_per_sample`](@ref)`(X, net)`: the per-sample log-density
 of `X` under the flow `net` conditioned on `Ctx`, as a vector of length `size(X, N)`.

 `Ctx` carries ONE context per sample -- its batch dimension matches `X`'s -- so a batch of
 different conditions is scored in a single pass. That is the case worth optimizing for: the
 caller that needs this is scoring a minibatch in which every entry has its own condition.

 Requires every log-determinant in `net` to be reportable per sample, and requires some layer
 in `net` to actually read the context: a chain that silently ignored it would return a
 perfectly ordinary unconditional density, and the only symptom would be a state pathway that
 never learns.

 See also: [`log_likelihood_per_sample`](@ref), [`forward_per_sample`](@ref)
"""
function log_likelihood_per_sample(X::AbstractArray{T,N}, Ctx::AbstractArray, net::Invertible;
                                   μ=T(0), σ=T(1), normalized::Bool=false,
                                   base=nothing) where {T,N}
    d = _latent_base(base, μ, σ, T)
    check_latent_support(net, d)
    _check_consumes_context(net)
    # Same setup side effect as the unconditional form, hidden from AD for the same reason.
    init!(net, X, Ctx)
    Z, logdet = flow_forward_per_sample(net, X, Ctx, parameter_data(net))
    return logpdf_per_sample(d, Z; normalized=normalized) .+ logdet
end

# Checked once per loss evaluation, at the entry point, rather than per layer per sample.
function _check_consumes_context(C::InvertibleChain)
    any(_consumes_context, C.layers) && return nothing
    throw(ArgumentError(
        "a context was supplied but no layer in this network reads one: build the coupling " *
        "layers with n_ctx = size(Ctx, ndims(Ctx)-1). Scoring would otherwise ignore the " *
        "context and report an unconditional density."))
end
_check_consumes_context(::Any) = nothing

# One pass per sample, for networks whose layers only report a batch-averaged
# log-determinant.
function _log_likelihood_per_sample(X::AbstractArray{T,N}, net::Invertible, ::Val{false},
                                    d, normalized::Bool) where {T,N}
    colons = ntuple(_ -> Colon(), Val(N-1))
    return [log_likelihood(X[colons..., i:i], net; base=d, normalized=normalized)
            for i in axes(X, N)]
end
