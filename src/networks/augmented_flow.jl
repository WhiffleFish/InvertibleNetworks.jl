# Augmented normalizing flows: a flow run on the data padded with auxiliary latent
# variables, so the network is no longer confined to the width of the data.
#
# Huang, Dinh and Courville (2020): https://arxiv.org/abs/2002.07101 (ANF)
# Chen et al. (2020): https://arxiv.org/abs/2002.09741 (VFlow)
# Nielsen et al. (2020): https://arxiv.org/abs/2007.02731 (SurVAE, augmentation as a surjection)

export AugmentedFlow, augmented_inverse, augmentation_size
export log_likelihood_importance, log_likelihood_importance_per_sample

"""
    A = AugmentedFlow(flow, naug; σ=1f0, logdet=true)

 Run `flow` on the data padded with `naug` extra channels of auxiliary noise, turning a
 bijection on the data space into a flow on the larger space `X × E`. The extra dimensions
 give the network width the data alone does not have, which is what an invertible
 architecture otherwise cannot buy: every intermediate representation of a plain flow has
 exactly the dimensionality of the data.

 The price is that the likelihood is no longer exact. `log p(x)` becomes a variational lower
 bound -- see the section on the bound below, which is worth reading before using this in
 anything that consumes densities quantitatively.

 *Input*:

 - `flow`: an invertible network accepting `n_channel + naug` channels, built with
   `logdet=true`

 - `naug`: number of auxiliary channels to append

 - `σ`: standard deviation of the auxiliary noise `ε ~ N(0, σ²I)`

 - `logdet`: whether the network reports the change-of-variables term

 *Usage:*

 - Forward mode: `Z, logdet = A.forward(X)` -- draws `ε`, appends it, and runs `flow`

 - Inverse mode: `X = A.inverse(Z)`, or `X, E = augmented_inverse(Z, A)` to keep the
   auxiliary part

 - Backward mode: `ΔX, X = A.backward(ΔZ, Z)`

 - Objective: `-log_likelihood(X, A)`, exactly as for an ordinary flow; the value is the
   evidence lower bound rather than the likelihood

 *Trainable parameters:*

 - The parameters of `flow`

# The bound

 With `ε ~ q(ε) = N(0, σ²I)` drawn independently of `x`, and `G` the map `flow` applies to
 the augmented input,

     log p(x) ≥ E_ε[ log p_Z(G(x, ε)) + log|det J_G(x, ε)| - log q(ε) ]

 the gap being `KL(q(ε) ‖ p(ε|x))`. `forward` folds the `-log q(ε)` term into the
 log-determinant it reports, so [`log_likelihood`](@ref) and
 [`log_likelihood_per_sample`](@ref) return this bound with no further bookkeeping, and the
 usual `-log_likelihood(X, A)` loss maximizes it.

 A *learned*, input-dependent `q(ε|x)` -- the difference between ANF/VFlow and plain noise
 padding -- needs no separate inference network here: make the first layer of `flow` a
 coupling layer that transforms the auxiliary channels conditioned on the data channels.
 Composing that with the fixed `N(0, σ²I)` draw gives `q(ε|x) = N(m(x), s(x)²)`, and its
 log-determinant is already accounted for. This is why `flow` is an ordinary flow on the
 wider space and not a special architecture.

# The bound only points the right way in the data-to-latent direction

 The bound above is an expectation over `ε ~ q`, which is what `forward` draws. The sampling
 direction is not the same computation: there `ε` falls out of `flow.inverse` alongside `x`,
 i.e. it is drawn from the model's own `p(ε|x)`, and

     E_{p(ε|x)}[log p(x, ε) - log q(ε)] = log p(x) + KL(p(ε|x) ‖ q(ε)) ≥ log p(x)

 which is an over-estimate, not a lower bound. So the score returned by
 [`inverse_and_log_likelihood`](@ref) for an `AugmentedFlow` is a stochastic estimate of
 `log p(x)` biased upward, and it is *not* interchangeable with the exact `log p` an
 unaugmented flow gives on that path. Anything that needs a calibrated density of its own
 samples -- an entropy bonus, an importance ratio between two policies -- will inherit that
 bias. Use an unaugmented flow where the exact density on the sampling path is what matters.

# Example

```julia
n_in, naug, n_hidden = 4, 4, 32
inner = InvertibleChain(ActNorm(n_in + naug; logdet=true),
                        CouplingLayerGlow(n_in + naug, n_hidden; logdet=true),
                        Conv1x1(n_in + naug),
                        CouplingLayerGlow(n_in + naug, n_hidden; logdet=true))
A = AugmentedFlow(inner, naug)

opt_state = Flux.setup(Adam(1f-3), A)
l, grads = Flux.withgradient(m -> -log_likelihood(X, m), A)   # maximizes the ELBO
Flux.update!(opt_state, A, grads[1])
```

 See also: [`log_likelihood_importance`](@ref), [`augmented_inverse`](@ref),
 [`InvertibleChain`](@ref)
"""
struct AugmentedFlow{F<:Invertible,T<:Real,LD} <: InvertibleNetwork
    flow::F
    naug::Int
    σ::T
    logdet::Bool
end

# As in `InvertibleChain`, `logdet` is carried as a type parameter so that `forward` infers
# whether it returns `Z` or `(Z, logdet)`; it is also a field so the generic `Invertible`
# machinery (and Functors reconstruction) sees a plain struct.
AugmentedFlow(flow::F, naug::Int, σ::T, logdet::Bool) where {F<:Invertible,T<:Real} =
    AugmentedFlow{F,T,logdet}(flow, naug, σ, logdet)

Flux.@layer AugmentedFlow

function AugmentedFlow(flow::Invertible, naug::Integer; σ=1f0, logdet::Bool=true)
    naug > 0 || throw(ArgumentError("naug must be positive, got $naug"))
    σ > 0 || throw(ArgumentError("σ must be positive, got $σ"))
    # Without the wrapped flow's log-determinant the bound is missing its main term, and the
    # failure is silent rather than loud: the network would happily report a number.
    if logdet && hasproperty(flow, :logdet) && !flow.logdet
        throw(ArgumentError(
            "AugmentedFlow needs the change-of-variables term of the flow it wraps, but " *
            "this $(nameof(typeof(flow))) does not accumulate one; build its layers with " *
            "logdet=true"))
    end
    return AugmentedFlow(flow, Int(naug), σ, logdet)
end

Base.show(io::IO, A::AugmentedFlow) =
    print(io, "AugmentedFlow(", typeof(A.flow).name.name, ", naug=", A.naug, ")")

"""
    n = augmentation_size(A)

 Number of auxiliary channels `A` appends to its input.
"""
augmentation_size(A::AugmentedFlow) = A.naug

# The augmented map is only as per-sample-capable as the flow it wraps; the `-log q(ε)` term
# is per sample by construction.
supports_per_sample_logdet(A::AugmentedFlow) = supports_per_sample_logdet(A.flow)


## Auxiliary noise and its density

# Channel dimension, which is also the dimension `tensor_cat`/`tensor_split` work along.
@inline _aug_dim(::Val{N}) where {N} = max(1, N - 1)

function _draw_augmentation(X::AbstractArray{T,N}, A::AugmentedFlow, rng) where {T,N}
    d = _aug_dim(Val(N))
    sz = ntuple(i -> i == d ? A.naug : size(X, i), Val(N))
    E = similar(X, T, sz)
    # `randn!` without an explicit generator is what dispatches to the GPU generator for a
    # `CuArray`; only reach for a user-supplied one when there is one.
    isnothing(rng) ? randn!(E) : randn!(rng, E)
    isone(A.σ) || (E .*= T(A.σ))
    return E
end

# log q(ε) per sample, normalizing constant included: the term is a genuine density, and
# leaving the constant out would make the reported bound depend on `naug` for no reason.
function _log_q_per_sample(A::AugmentedFlow, E::AbstractArray{T,N}) where {T,N}
    d = length(E) ÷ size(E, N)
    return per_sample_sum(-T(.5)*(E/T(A.σ)).^2) .- T(d/2*log(2*π*A.σ^2))
end

_log_q(A::AugmentedFlow, E::AbstractArray{T,N}) where {T,N} =
    sum(_log_q_per_sample(A, E))/size(E, N)

# Split an augmented tensor back into its data and auxiliary parts.
_split_augmented(A::AugmentedFlow, XA::AbstractArray{T,N}) where {T,N} =
    tensor_split(XA; split_index=size(XA, _aug_dim(Val(N))) - A.naug)


## Forward/inverse/backward

"""
    Z, logdet = forward(X, A::AugmentedFlow)
    Z, logdet = forward(X, A::AugmentedFlow; logdet=:sample)

 Draw `ε ~ N(0, σ²I)`, append it to `X` along the channel dimension, and run the wrapped
 flow. `logdet` is the flow's log-determinant less `log q(ε)`, so that
 `log_likelihood(Z) + logdet` is the evidence lower bound on `log p(X)`.

 Unlike every other `forward` in this package this one is stochastic: two calls on the same
 `X` draw different `ε` and give different `Z`. Pass `ε` explicitly to pin the draw down, and
 `rng` to control the generator.

 With `logdet=:sample` the bound is returned per sample rather than averaged over the batch.
"""
function forward(X::AbstractArray{T,N}, A::AugmentedFlow{F,S,LD}; logdet=nothing, rng=nothing,
                 ε=nothing) where {T,N,F,S,LD}
    E = isnothing(ε) ? _draw_augmentation(X, A, rng) : ε
    return _augmented_forward(X, E, A, logdet_mode(logdet, Val(LD)))
end

function _augmented_forward(X::AbstractArray{T,N}, E, A::AugmentedFlow, ::Val{false}) where {T,N}
    out = forward(tensor_cat(X, E), A.flow; logdet=false)
    return out isa Tuple ? out[1] : out
end

function _augmented_forward(X::AbstractArray{T,N}, E, A::AugmentedFlow, ::Val{true}) where {T,N}
    Z, ld = _flow_forward_checked(tensor_cat(X, E), A.flow, true)
    return Z, ld - _log_q(A, E)
end

function _augmented_forward(X::AbstractArray{T,N}, E, A::AugmentedFlow, ::Val{:sample}) where {T,N}
    Z, ld = _flow_forward_checked(tensor_cat(X, E), A.flow, :sample)
    return Z, ld .- _log_q_per_sample(A, E)
end

function _flow_forward_checked(XA::AbstractArray, flow::Invertible, mode)
    out = forward(XA, flow; logdet=mode)
    out isa Tuple || throw(ArgumentError(
        "AugmentedFlow needs the change-of-variables term of the flow it wraps, but " *
        "$(nameof(typeof(flow))) does not accumulate a log-determinant; build its layers " *
        "with logdet=true"))
    return out
end

"""
    X = inverse(Z, A::AugmentedFlow)
    X, logdet = inverse(Z, A::AugmentedFlow; logdet=true)

 Invert the wrapped flow and drop the auxiliary channels, giving a sample in data space.

 With a log-determinant requested, the term returned is such that
 `log_likelihood(Z) - logdet` is the joint score `log p(x, ε) - log q(ε)` at the `ε` that
 came out of the inverse pass. That quantity over-estimates `log p(x)` in expectation -- see
 the note in [`AugmentedFlow`](@ref) -- so treat it as a diagnostic, not as a density.

 Use [`augmented_inverse`](@ref) to keep `ε`.
"""
function inverse(Z::AbstractArray{T,N}, A::AugmentedFlow; logdet=false) where {T,N}
    out = _augmented_inverse(Z, A, logdet_mode(logdet))
    return length(out) == 2 ? out[1] : (out[1], out[3])
end

"""
    X, E = augmented_inverse(Z, A::AugmentedFlow)
    X, E, logdet = augmented_inverse(Z, A::AugmentedFlow; logdet=true)

 Like [`inverse`](@ref), but returns the auxiliary part `E` alongside the data part instead
 of discarding it. Feeding both back through `forward(X, A; ε=E)` reproduces `Z`, which
 `A.forward(X)` on its own cannot do: it would draw a fresh `ε`.
"""
augmented_inverse(Z::AbstractArray{T,N}, A::AugmentedFlow; logdet=false) where {T,N} =
    _augmented_inverse(Z, A, logdet_mode(logdet))

function _augmented_inverse(Z::AbstractArray{T,N}, A::AugmentedFlow, ::Val{false}) where {T,N}
    out = inverse(Z, A.flow)
    XA = out isa Tuple ? out[1] : out
    return _split_augmented(A, XA)
end

function _augmented_inverse(Z::AbstractArray{T,N}, A::AugmentedFlow, mode::Val) where {T,N}
    XA, ld = _flow_inverse_checked(Z, A.flow, _mode_kwarg(mode))
    X, E = _split_augmented(A, XA)
    # `forward` reports `ld_forward - log q(ε)`; the inverse term is its negative.
    return X, E, _inverse_logdet(ld, A, E, mode)
end

@inline _inverse_logdet(ld, A::AugmentedFlow, E, ::Val{true}) = ld + _log_q(A, E)
@inline _inverse_logdet(ld, A::AugmentedFlow, E, ::Val{:sample}) = ld .+ _log_q_per_sample(A, E)

function _flow_inverse_checked(Z::AbstractArray, flow::Invertible, mode)
    out = inverse(Z, flow; logdet=mode)
    out isa Tuple || throw(ArgumentError(
        "AugmentedFlow needs the change-of-variables term of the flow it wraps, but " *
        "$(nameof(typeof(flow))) does not accumulate a log-determinant; build its layers " *
        "with logdet=true"))
    return out
end

# The auxiliary noise is drawn independently of the parameters and of `X`, so `-log q(ε)`
# contributes nothing to either gradient: the whole backward pass is the wrapped flow's,
# with the auxiliary slice of the input cotangent dropped at the boundary.
function backward(ΔZ::AbstractArray{T,N}, Z::AbstractArray{T,N}, A::AugmentedFlow;
                  set_grad::Bool=true, logdet_weight=nothing) where {T,N}
    set_grad || throw(ArgumentError("AugmentedFlow only implements backward with " *
                                    "set_grad=true; use the wrapped flow directly for the " *
                                    "Jacobian interface"))
    ΔXA, XA = isnothing(logdet_weight) ? backward(ΔZ, Z, A.flow) :
                                         backward(ΔZ, Z, A.flow; logdet_weight=logdet_weight)
    ΔX, _ = _split_augmented(A, ΔXA)
    X, _ = _split_augmented(A, XA)
    return ΔX, X
end

function jacobian(::AbstractArray{T,N}, ::AbstractVector{<:Parameter}, ::AbstractArray{T,N},
                  ::AugmentedFlow) where {T,N}
    throw(ArgumentError("Jacobian for AugmentedFlow is not defined: the map is stochastic " *
                        "in its auxiliary variables; take the Jacobian of the wrapped flow"))
end

# Same AD entry point as `InvertibleChain`: `A(X)` returns `(Z, logdet)` inside a gradient
# and outside it alike.
(A::AugmentedFlow)(X::AbstractArray) = flow_forward(A, X, parameter_data(A))


## Importance-weighted bound

"""
    f = log_likelihood_importance(X, A::AugmentedFlow; nsamples=8, μ=0f0, σ=1f0,
                                  normalized=false)

 Importance-weighted lower bound on `log p(X)` under an [`AugmentedFlow`](@ref), using
 `nsamples` independent draws of the auxiliary variable:

     log p(x) ≥ E[ log (1/K) Σₖ p(x, εₖ)/q(εₖ) ]

 This is the multi-sample tightening of the bound [`log_likelihood`](@ref) returns, and is
 what to report when the number is meant to be read as a density rather than only
 differentiated: the single-sample bound understates `log p(x)` by the full
 `KL(q(ε) ‖ p(ε|x))`, and the gap shrinks monotonically in `nsamples`. `nsamples=1` recovers
 the plain ELBO in expectation.

 Averaged over the batch and differentiable, following the conventions of
 [`log_likelihood`](@ref). Costs `nsamples` forward passes.

 See also: [`log_likelihood_importance_per_sample`](@ref), [`log_likelihood`](@ref)
"""
function log_likelihood_importance(X::AbstractArray{T,N}, A::AugmentedFlow; nsamples::Int=8,
                                   μ=T(0), σ=T(1), normalized::Bool=false) where {T,N}
    f = log_likelihood_importance_per_sample(X, A; nsamples=nsamples, μ=μ, σ=σ,
                                             normalized=normalized)
    return sum(f)/size(X, N)
end

"""
    f = log_likelihood_importance_per_sample(X, A::AugmentedFlow; nsamples=8, μ=0f0, σ=1f0,
                                             normalized=false)

 Per-sample form of [`log_likelihood_importance`](@ref): a vector of length `size(X, N)`
 holding the importance-weighted bound for each sample.
"""
function log_likelihood_importance_per_sample(X::AbstractArray{T,N}, A::AugmentedFlow;
                                              nsamples::Int=8, μ=T(0), σ=T(1),
                                              normalized::Bool=false) where {T,N}
    nsamples > 0 || throw(ArgumentError("nsamples must be positive, got $nsamples"))
    init!(A, X)
    scores = map(1:nsamples) do _
        Z, logdet = forward_per_sample(X, A)
        log_likelihood_per_sample(Z; μ=μ, σ=σ, normalized=normalized) .+ logdet
    end
    return _logmeanexp(scores)
end

# log (1/K) Σₖ exp(sₖ), stabilized, over a vector of per-sample score vectors. Written
# without mutation so Zygote can differentiate through it.
function _logmeanexp(scores::AbstractVector{<:AbstractVector{T}}) where {T}
    K = length(scores)
    K == 1 && return scores[1]
    m = reduce((a, b) -> max.(a, b), scores)
    total = reduce(+, map(s -> exp.(s .- m), scores))
    return m .+ log.(total) .- T(log(K))
end
