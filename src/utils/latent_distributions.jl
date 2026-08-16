# Pluggable base (latent) distributions for normalizing flows, with a domain invariant.
#
# Change of variables does not care which base distribution a flow pushes forward:
#
#     log p_X(x) = log p_Z(f(x)) + log|det J_f(x)|
#
# Gaussian is a convention, not a constraint. What *is* a constraint -- and what a bare
# `base=` keyword would silently violate -- is that the network has to be a bijection of the
# base's support. With a Gaussian that is free, since every layer here is a bijection of
# `Rᵈ`. With a base supported on a box it is not: a `Conv1x1` rotates points out of the box,
# an `ActNorm` shifts them out, an affine coupling scales them out. The density such a
# combination reports is wrong wherever the flow leaves the box, and wrong *quietly* -- no
# error, no NaN. So the support travels with the distribution and is checked against the
# network before any likelihood is computed.

export LatentDistribution, StandardNormal, BoxUniform
export logpdf_per_sample, logpdf_mean, latent_support, uniform_lognorm
export UnboundedSupport, BoxSupport
export preserves_box, box_violation, check_latent_support


###################################################################################################
# Supports

"""
    UnboundedSupport()

Support of a latent distribution that covers all of `Rᵈ`. Any bijection of `Rᵈ` -- which is
every layer in this package -- may be used with such a base.

See also: [`BoxSupport`](@ref), [`latent_support`](@ref)
"""
struct UnboundedSupport end

"""
    BoxSupport(B)

Support of a latent distribution confined to the box `[-B, B]ᵈ`. A network scored against
such a base must be a bijection *of that box*; see [`preserves_box`](@ref).

See also: [`UnboundedSupport`](@ref), [`latent_support`](@ref)
"""
struct BoxSupport{T<:Real}
    B::T
end

Base.show(io::IO, ::UnboundedSupport) = print(io, "UnboundedSupport()")
Base.show(io::IO, s::BoxSupport) = print(io, "BoxSupport(", s.B, ")")


###################################################################################################
# The interface

"""
    LatentDistribution

Base (latent) distribution of a normalizing flow. Implementations provide

- [`logpdf_per_sample`](@ref)`(d, Z; normalized=false)` -- log-density of each sample of a
  batched array `Z`, as a vector of length `size(Z, N)`
- [`latent_support`](@ref)`(d)` -- [`UnboundedSupport`](@ref) or [`BoxSupport`](@ref)
- `rand(d, dims...)` -- a draw, replacing the implicit `randn` a sampling loop would write

and may specialize [`logpdf_mean`](@ref) when the batch average has a cheaper form than
averaging the per-sample vector.

Every log-likelihood entry point takes one as the `base` keyword:

```julia
log_likelihood(X, flow; base=StandardNormal())          # the default
log_likelihood(X, flow; base=BoxUniform(3f0))           # uniform on [-3, 3]ᵈ
```

See also: [`StandardNormal`](@ref), [`BoxUniform`](@ref), [`log_likelihood`](@ref)
"""
abstract type LatentDistribution end

"""
    v = logpdf_per_sample(d::LatentDistribution, Z; normalized=false)

Log-density of `d` at each sample of the batched array `Z`, as a vector of length
`size(Z, N)`.

Following the convention of [`log_likelihood`](@ref), the normalizing constant is dropped by
default, so the result is a log-density up to an additive constant -- enough for
optimization, since the constant does not depend on the model. Pass `normalized=true` for a
calibrated log-density.

See also: [`logpdf_mean`](@ref), [`log_likelihood_per_sample`](@ref)
"""
function logpdf_per_sample end

"""
    f = logpdf_mean(d::LatentDistribution, Z; normalized=false)

Log-density of `d` on `Z`, averaged over the batch: the aggregate form of
[`logpdf_per_sample`](@ref), matching the convention of [`log_likelihood`](@ref).

The generic method averages the per-sample vector; implementations may specialize it when a
direct reduction is cheaper.
"""
logpdf_mean(d::LatentDistribution, Z::AbstractArray{T,N}; normalized::Bool=false) where {T,N} =
    sum(logpdf_per_sample(d, Z; normalized=normalized))/size(Z, N)

"""
    s = latent_support(d::LatentDistribution)

Region of `Rᵈ` on which `d` puts its mass: [`UnboundedSupport`](@ref) or
[`BoxSupport`](@ref)`(B)`. A flow scored against a bounded base must be a bijection of that
region -- see [`check_latent_support`](@ref).
"""
function latent_support end

# Number of elements in one sample of a batched array.
@inline _sample_dim(X::AbstractArray{T,N}) where {T,N} = length(X) ÷ size(X, N)


###################################################################################################
# Gaussian

"""
    d = StandardNormal(μ=0f0, σ=1f0)

Gaussian base distribution with mean `μ` and standard deviation `σ`, iid over the elements of
a sample. This is the default base of every likelihood entry point, and reproduces exactly
what the `μ`/`σ` keywords have always done.

Its support is all of `Rᵈ`, so any network in this package may be scored against it.

See also: [`LatentDistribution`](@ref), [`BoxUniform`](@ref)
"""
struct StandardNormal{T<:Real} <: LatentDistribution
    μ::T
    σ::T

    # Inner, so that the default constructors -- which would bypass the check -- are not
    # generated alongside it.
    function StandardNormal{T}(μ::T, σ::T) where {T<:Real}
        σ > 0 || throw(ArgumentError("σ must be positive, got σ = $σ"))
        return new{T}(μ, σ)
    end
end

StandardNormal(μ::T, σ::T) where {T<:Real} = StandardNormal{T}(μ, σ)
StandardNormal(μ::Real, σ::Real) = StandardNormal(promote(μ, σ)...)
StandardNormal(μ::Real) = StandardNormal(μ, one(μ))
StandardNormal() = StandardNormal(0f0, 1f0)

Base.show(io::IO, d::StandardNormal) = print(io, "StandardNormal(", d.μ, ", ", d.σ, ")")

latent_support(::StandardNormal) = UnboundedSupport()

# The quadratic kernel, elementwise, at the element type of the data.
@inline _gauss_kernel(d::StandardNormal, X::AbstractArray{T}) where {T} =
    -T(.5)*((X .- T(d.μ))/T(d.σ)).^2

function logpdf_per_sample(d::StandardNormal, X::AbstractArray{T,N};
                           normalized::Bool=false) where {T,N}
    f = per_sample_sum(_gauss_kernel(d, X))
    return normalized ? f .+ gaussian_lognorm(X, T(d.σ)) : f
end

# Reducing over the batch as well is one pass, not a per-sample vector plus a second sum.
logpdf_mean(d::StandardNormal, X::AbstractArray{T,N}; normalized::Bool=false) where {T,N} =
    T(1/size(X, N))*sum(_gauss_kernel(d, X)) +
    (normalized ? gaussian_lognorm(X, T(d.σ)) : zero(T))


###################################################################################################
# Uniform on a box

"""
    d = BoxUniform(B)

Uniform base distribution on the box `[-B, B]ᵈ`, with log-density `-d·log(2B)` inside and
`-Inf` outside.

There is no uniform distribution on `Rᵈ`, so this is inherently a bounded-domain base: the
network scored against it must be a bijection of `[-B, B]ᵈ`, which in this package means
spline layers with `mix=false` and a bound no larger than `B`. Anything else is rejected at
the likelihood call rather than silently producing wrong numbers -- see
[`check_latent_support`](@ref).

# Why it is worth the constraint

A rational-quadratic spline with linear tails is already a bijection of `[-B, B]` onto
itself, so a stack of `CouplingLayerSpline(...; mix=false)` is a bijection of `[-B, B]ᵈ`. Pair
it with this base and the two constants of the change of variables cancel: an
`identity_init` chain is *exactly* uniform, at Float32 precision and at every parameter
value, with no squashing layer and none of its boundary failure modes. The usual alternative
-- a Gaussian base on `Rᵈ` squashed into the box by a sigmoid -- can only approximate a
uniform, and loses precision near the boundary.

# Example

```julia
flow = InvertibleChain(CouplingLayerSpline(n, n_hidden; bound=3f0, mix=false, logdet=true),
                       CouplingLayerSpline(n, n_hidden; bound=3f0, mix=false, swap=true,
                                           logdet=true))
log_likelihood(X, flow; base=BoxUniform(3f0), normalized=true)
```

See also: [`LatentDistribution`](@ref), [`StandardNormal`](@ref), [`preserves_box`](@ref)
"""
struct BoxUniform{T<:Real} <: LatentDistribution
    B::T

    function BoxUniform{T}(B::T) where {T<:Real}
        B > 0 || throw(ArgumentError("the box half-width must be positive, got B = $B"))
        return new{T}(B)
    end
end

BoxUniform(B::T) where {T<:Real} = BoxUniform{T}(B)

Base.show(io::IO, d::BoxUniform) = print(io, "BoxUniform(", d.B, ")")

latent_support(d::BoxUniform) = BoxSupport(d.B)

"""
    c = uniform_lognorm(X, B)

Per-sample log-density of the uniform distribution on `[-B, B]ᵈ`, `-d·log(2B)`, where
`d = length(X) ÷ size(X, N)` is the number of elements in one sample of the batched array `X`.

The uniform analogue of [`gaussian_lognorm`](@ref): it *is* the whole log-density, since the
kernel is flat, which is why `normalized=false` leaves nothing behind for this base.
"""
uniform_lognorm(X::AbstractArray{T,N}, B) where {T,N} = T(-_sample_dim(X)*log(2*B))

# `-Inf` on the samples that fall outside the box, `0` on the rest. Piecewise constant in `Z`,
# so its derivative is zero wherever it exists; hiding it from AD says that directly instead
# of letting a `-Inf` branch reach a pullback and come back as a NaN.
function _outside_box_penalty(Z::AbstractArray{T,N}, B) where {T,N}
    return per_sample_sum(ifelse.(abs.(Z) .<= T(B), zero(T), T(-Inf)))
end

@non_differentiable _outside_box_penalty(::Any, ::Any)

function logpdf_per_sample(d::BoxUniform, Z::AbstractArray{T,N};
                           normalized::Bool=false) where {T,N}
    f = _outside_box_penalty(Z, d.B)
    return normalized ? f .+ uniform_lognorm(Z, d.B) : f
end

logpdf_mean(d::BoxUniform, Z::AbstractArray{T,N}; normalized::Bool=false) where {T,N} =
    sum(_outside_box_penalty(Z, d.B))/size(Z, N) +
    (normalized ? uniform_lognorm(Z, d.B) : zero(T))


###################################################################################################
# Sampling
#
# `randn(Float32, ...)` in a sampling loop is the second place the Gaussian assumption is
# baked in -- implicitly, and in user code rather than package code. Going through the
# distribution keeps a change of base a one-line change.

"""
    Z = rand([rng], d::LatentDistribution, dims...)

Draw an array of latent samples from `d`, of element type `eltype(d)`. This is what a
generative pass should feed to `inverse`: writing `randn(Float32, dims...)` there hard-codes
a Gaussian base in user code, where a `base` keyword cannot reach it.

# Example

```julia
Z = rand(base, nx, ny, n, batchsize)
X, logp = inverse_and_log_likelihood(Z, flow; base=base)
```
"""
Base.rand(rng::AbstractRNG, d::StandardNormal{T}, dims::Integer...) where {T} =
    T(d.μ) .+ T(d.σ) .* randn(rng, T, dims...)

Base.rand(rng::AbstractRNG, d::BoxUniform{T}, dims::Integer...) where {T} =
    T(2*d.B) .* rand(rng, T, dims...) .- T(d.B)

Base.rand(d::LatentDistribution, dims::Integer...) = rand(Random.default_rng(), d, dims...)
Base.rand(rng::AbstractRNG, d::LatentDistribution, dims::Dims) = rand(rng, d, dims...)
Base.rand(d::LatentDistribution, dims::Dims) = rand(Random.default_rng(), d, dims...)

Base.eltype(::Type{<:StandardNormal{T}}) where {T} = T
Base.eltype(::Type{<:BoxUniform{T}}) where {T} = T
Base.eltype(d::LatentDistribution) = eltype(typeof(d))


###################################################################################################
# The domain invariant
#
# `preserves_box(x, B)` answers "is `x` a bijection of `[-B, B]ᵈ`?", conservatively: unless a
# layer says otherwise it is assumed not to be, so a new layer type cannot silently opt into a
# bounded base. `box_violation` returns the same answer with an explanation attached, and is
# only consulted on the failure path, so the check itself allocates nothing.
#
# The predicate takes the bound rather than being a plain per-type trait because whether a
# spline layer preserves a box depends on *which* box: a spline with half-interval `B_spline`
# is the identity outside `[-B_spline, B_spline]` and a bijection inside it, so it is a
# bijection of `[-B, B]` exactly when `B ≥ B_spline`. Nesting the bounds the wrong way round
# is easy to do and impossible to see in the output, so it is the same check.

"""
    b = preserves_box(x, B)

Whether `x` -- a layer or a network -- is a bijection of the box `[-B, B]ᵈ`, i.e. whether it
maps that box onto itself in both directions.

Defaults to `false`: a layer has to say that it preserves a bounded domain, so that a layer
type this trait has never heard of cannot silently be paired with a [`BoxUniform`](@ref) base.

The bound is an argument rather than the trait being per-type because for spline layers the
answer depends on it. A spline with half-interval `B_spline` is the identity outside
`[-B_spline, B_spline]`, so it is a bijection of `[-B, B]` exactly when `B ≥ B_spline`; with
the bounds nested the other way a point inside the base's box can be mapped out of it. For
`CouplingLayerSpline` it is also per *instance* rather than per type: `mix=true` puts a
[`Conv1x1`](@ref) in front, whose rotation leaves the box.

See also: [`box_violation`](@ref), [`check_latent_support`](@ref), [`BoxUniform`](@ref)
"""
preserves_box(::Any, B) = false

"""
    msg = box_violation(x, B)

`nothing` when [`preserves_box`](@ref)`(x, B)`, and otherwise a description of why `x` is not
a bijection of `[-B, B]ᵈ`, suitable for an error message. For a network, the descriptions of
all its offending layers.

Consulted only on the failure path, so the domain check costs nothing when it passes.
"""
box_violation(x::Any, B) =
    "$(nameof(typeof(x))) is not a bijection of [-$B, $B]ᵈ"

"""
    check_latent_support(net, base::LatentDistribution)

Verify that `net` is a bijection of the support of `base`, throwing an `ArgumentError` naming
the offending layers if it is not. A no-op for an unbounded base such as
[`StandardNormal`](@ref), which every network in this package is compatible with.

Called by every likelihood entry point that takes both a network and a `base`. Call it
directly to fail at model-construction time rather than at the first likelihood evaluation.

# Example

```julia
flow = InvertibleChain(CouplingLayerSpline(n, nh; bound=3f0, mix=false, logdet=true))
check_latent_support(flow, BoxUniform(3f0))    # ok
check_latent_support(flow, BoxUniform(1f0))    # errors: the base's box is inside the spline's
```

See also: [`preserves_box`](@ref), [`BoxUniform`](@ref)
"""
check_latent_support(net, base::LatentDistribution) =
    _check_latent_support(net, latent_support(base), base)

_check_latent_support(::Any, ::UnboundedSupport, ::LatentDistribution) = nothing

function _check_latent_support(net, s::BoxSupport, base::LatentDistribution)
    preserves_box(net, s.B) && return nothing
    throw(ArgumentError(
        "base $base is supported on [-$(s.B), $(s.B)]ᵈ, but this network is not a bijection " *
        "of that box: $(box_violation(net, s.B)). Change of variables needs the flow to map " *
        "the base's support onto itself; otherwise the density is wrong wherever it does " *
        "not. Use spline layers with mix=false and bound <= $(s.B), or a base with " *
        "unbounded support such as StandardNormal()."))
end


###################################################################################################
# Resolving the `base` keyword against the legacy `μ`/`σ` ones

# `μ`/`σ` are the Gaussian-only shorthand the package has always had; `base` is the general
# form. Both at once is a contradiction rather than a precedence question, so it is an error
# instead of a silently ignored argument.
_latent_base(base, μ, σ, ::Type) = throw(ArgumentError(
    "`base` must be a LatentDistribution, got $(typeof(base))"))

_latent_base(::Nothing, μ, σ, ::Type{T}) where {T} = StandardNormal(T(μ), T(σ))

function _latent_base(base::LatentDistribution, μ, σ, ::Type)
    (μ == 0 && σ == 1) || throw(ArgumentError(
        "pass either `base` or `μ`/`σ`, not both: base = $base already fixes the latent " *
        "distribution, so μ = $μ, σ = $σ would be ignored"))
    return base
end

# The closed-form gradient and Hessian of the log-likelihood are Gaussian-specific; the AD
# path does not use them, so they stay Gaussian-only rather than growing an interface method
# every base would have to implement.
_gaussian_base(base, μ, σ, ::Type{T}) where {T} = _gaussian_only(_latent_base(base, μ, σ, T))
_gaussian_only(d::StandardNormal) = d
_gaussian_only(d::LatentDistribution) = throw(ArgumentError(
    "∇log_likelihood/Hlog_likelihood are closed forms for a Gaussian base and have no " *
    "definition for $d; differentiate log_likelihood with Zygote instead, which works for " *
    "any base"))


###################################################################################################
# Which layers preserve a box
#
# Kept together rather than spread over the layer files: the interesting content is the
# comparison between them, and the list is short.

# A spline is the identity outside its own half-interval and a bijection inside it, so it is a
# bijection of any box that contains that interval. Compared with a tolerance because the
# bound is stored as a Float32 and the base's may not be.
@inline _box_covers(B, bound) = B > bound || B ≈ bound

preserves_box(s::SplineSpec, B) = _box_covers(B, s.bound)
# A circular spline is a bijection of the *torus* `[-B, B)`: input outside it is wrapped in,
# which is a projection, not a bijection. So the two bounds have to agree exactly.
preserves_box(s::SplineSpec{:circular}, B) = B ≈ s.bound

_spline_box_violation(s::SplineSpec, B, name) =
    preserves_box(s, B) ? nothing :
    "$name has spline bound $(s.bound) > $B, so a point inside [-$B, $B] can be mapped " *
    "outside it -- the spline is only the identity beyond its own bound (needs bound <= $B)"

_spline_box_violation(s::SplineSpec{:circular}, B, name) =
    preserves_box(s, B) ? nothing :
    "$name is a bijection of the torus [-$(s.bound), $(s.bound)), not of [-$B, $B]; a " *
    "circular spline needs a base whose bound equals its own"

preserves_box(L::SplineLayer, B) = preserves_box(L.spline, B)
box_violation(L::SplineLayer, B) = _spline_box_violation(L.spline, B, "SplineLayer")

preserves_box(L::CouplingLayerSpline, B) = isnothing(L.C) && preserves_box(L.spline, B)

function box_violation(L::CouplingLayerSpline, B)
    isnothing(L.C) || return "CouplingLayerSpline(mix=true) puts a Conv1x1 in front, whose " *
                             "rotation maps points out of [-$B, $B]ᵈ (use mix=false and " *
                             "alternate `swap` between layers instead)"
    return _spline_box_violation(L.spline, B, "CouplingLayerSpline")
end

# The layers that definitely do not, with the reason spelled out: these are the ones a user
# reaching for a bounded base is most likely to have in their chain already.
box_violation(::ActNorm, B) =
    "ActNorm applies a per-channel shift and scale, which moves points out of [-$B, $B]ᵈ"
box_violation(::Conv1x1, B) =
    "Conv1x1 is a rotation, which does not preserve [-$B, $B]ᵈ"
box_violation(::CouplingLayerGlow, B) =
    "CouplingLayerGlow applies an unbounded affine map s .* x .+ t to half its channels"
box_violation(::AffineLayer, B) =
    "AffineLayer applies an unbounded elementwise affine map"
box_violation(L::BoundedBijector, B) =
    "BoundedBijector maps ($(L.low), $(L.high)) onto all of R, which is not [-$B, $B]"

# A chain is a bijection of the box exactly when every one of its layers is.
preserves_box(C::InvertibleChain, B) = all(l -> preserves_box(l, B), C.layers)

function box_violation(C::InvertibleChain, B)
    reasons = [box_violation(l, B) for l in C.layers if !preserves_box(l, B)]
    return isempty(reasons) ? nothing : join(reasons, "; ")
end

# Reversing a bijection of the box gives a bijection of the box.
preserves_box(R::Reversed, B) = preserves_box(R.I, B)
box_violation(R::Reversed, B) = box_violation(R.I, B)

# Augmentation appends Gaussian noise, which is not confined to any box, and the flow it
# wraps runs on the widened space.
box_violation(::AugmentedFlow, B) =
    "AugmentedFlow appends N(0, σ²) auxiliary channels, which are not confined to " *
    "[-$B, $B]; a bounded base has no meaning on the augmented space"
