# Monotonic spline transforms: the elementwise maps behind neural spline flows.
#
#   rational-quadratic  (`:rqs`)      -- Durkan et al. (2019),     https://arxiv.org/abs/1906.04032
#   circular / periodic (`:circular`) -- Rezende et al. (2020),    https://arxiv.org/abs/2002.02428
#   linear rational     (`:lrs`)      -- Dolatabadi et al. (2020), https://arxiv.org/abs/2001.05168
#
# A spline transform replaces the affine `y = s*x + t` of a coupling layer with a monotonic
# piecewise map whose knots are predicted by the conditioner. It is strictly more expressive
# per layer -- one coupling layer can fit a multi-modal conditional -- while keeping the
# analytic single-pass inverse that an affine flow buys and an autoregressive flow gives up.
#
# Everything here is elementwise in the value being transformed and couples only across the
# *bin* axis, so the whole file is written as broadcasts over arrays carrying one extra
# dimension of length `K` (the bins). The layers in `layers/invertible_layer_spline.jl`
# arrange for that axis to sit at dimension `D`, immediately after the channel dimension, so
# that the conditioner's output tensor reshapes into place without a `permutedims`.
#
# Gradients are hand-derived, like the rest of this package: `spline_vjp` returns the residual
# with respect to the input and to the *raw* parameter tensor -- the conditioner's output,
# before the softmax/softplus normalization -- so a coupling layer can hand it straight to
# `block_backward`.

export SplineSpec, n_spline_params

const SPLINE_KINDS = (:rqs, :circular, :lrs)

"""
    spec = SplineSpec(kind=:rqs; nbins=8, bound=3f0, min_bin_width=1f-3,
                      min_bin_height=1f-3, min_derivative=1f-3, min_lambda=1f-3,
                      identity_init=true)

 Shape of a monotonic spline transform: how many bins, over what interval, and which family.
 Carried by [`CouplingLayerSpline`](@ref) and [`SplineLayer`](@ref); it holds no trainable
 state of its own.

 *Input*:

 - `kind`: `:rqs` for monotonic rational-quadratic splines with linear tails (the transform of
   Durkan et al., 2019), `:circular` for the periodic variant of Rezende et al. (2020), or
   `:lrs` for the linear rational splines of Dolatabadi et al. (2020)

 - `nbins`: number of bins `K` spanning the spline interval. More bins means a more flexible
   map at a cost linear in conditioner outputs; the paper uses 8 for density estimation and up
   to 128 for two-dimensional toy problems

 - `bound`: half-width `B` of the interval `[-B, B]` the spline acts on. For `:rqs` and `:lrs`
   the map is the identity outside it, so `B` should comfortably contain the data; the paper
   reports little sensitivity over `B ∈ [1, 5]`. For `:circular` it is instead the half-period,
   and inputs are wrapped into `[-B, B)`

 - `min_bin_width`, `min_bin_height`, `min_derivative`, `min_lambda`: floors that keep bins
   from collapsing and derivatives from reaching zero, which is what keeps the map invertible
   in finite precision

 - `identity_init`: scale the softplus on the derivatives so that all-zero parameters give
   exactly the identity map. A layer built this way starts as a no-op, which is the usual
   reason a deep flow trains stably from the first step

 See also: [`CouplingLayerSpline`](@ref), [`SplineLayer`](@ref), [`n_spline_params`](@ref)
"""
struct SplineSpec{K}
    nbins::Int
    bound::Float32
    min_bin_width::Float32
    min_bin_height::Float32
    min_derivative::Float32
    min_lambda::Float32
    identity_init::Bool
end

function SplineSpec(kind::Symbol=:rqs; nbins::Int=8, bound=3f0, min_bin_width=1f-3,
                    min_bin_height=1f-3, min_derivative=1f-3, min_lambda=1f-3,
                    identity_init::Bool=true)
    kind in SPLINE_KINDS || throw(ArgumentError(
        "unknown spline kind :$kind, expected one of $(SPLINE_KINDS)"))
    nbins ≥ 2 || throw(ArgumentError("a spline needs at least 2 bins, got nbins = $nbins"))
    bound > 0 || throw(ArgumentError("the spline bound must be positive, got bound = $bound"))
    nbins*min_bin_width < 1 || throw(ArgumentError(
        "nbins*min_bin_width = $(nbins*min_bin_width) leaves no width to distribute; " *
        "lower min_bin_width or nbins"))
    nbins*min_bin_height < 1 || throw(ArgumentError(
        "nbins*min_bin_height = $(nbins*min_bin_height) leaves no height to distribute; " *
        "lower min_bin_height or nbins"))
    0 ≤ min_derivative < 1 || throw(ArgumentError(
        "need 0 <= min_derivative < 1, got $min_derivative"))
    0 ≤ min_lambda < 0.5 || throw(ArgumentError("need 0 <= min_lambda < 0.5, got $min_lambda"))
    return SplineSpec{kind}(nbins, Float32(bound), Float32(min_bin_width),
                            Float32(min_bin_height), Float32(min_derivative),
                            Float32(min_lambda), identity_init)
end

Base.show(io::IO, s::SplineSpec{K}) where {K} =
    print(io, "SplineSpec(:$K; nbins=$(s.nbins), bound=$(s.bound))")

# Configuration, not state: it holds no arrays, so `Flux.f32`/`f64`/`gpu` have nothing to
# convert here. Saying so explicitly also stops them trying to rebuild it -- the spline kind
# lives in the type parameter, which the field values alone cannot recover.
Flux.Functors.@leaf SplineSpec

# The layers read the interval and the floors at the element type of their input, so a spec
# shared by a Float32 and a Float64 model still behaves.

"""
    P = n_spline_params(spec)

 Number of conditioner outputs one spline needs per transformed value.

 `3K-1` for `:rqs` (K widths, K heights, K-1 internal derivatives -- the two boundary
 derivatives are pinned to 1 so the map joins its linear tails smoothly), `3K` for `:circular`
 (the boundary derivative is shared and learned rather than pinned, which is what makes the
 map a diffeomorphism of the circle), and `4K-1` for `:lrs` (an extra interpolation weight `λ`
 per bin).

 See also: [`SplineSpec`](@ref)
"""
n_spline_params(s::SplineSpec{:rqs})      = 3*s.nbins - 1
n_spline_params(s::SplineSpec{:circular}) = 3*s.nbins
n_spline_params(s::SplineSpec{:lrs})      = 4*s.nbins - 1

# Number of stored derivative parameters, before the boundary values are attached.
n_deriv_params(s::SplineSpec{:rqs})      = s.nbins - 1
n_deriv_params(s::SplineSpec{:circular}) = s.nbins
n_deriv_params(s::SplineSpec{:lrs})      = s.nbins - 1

has_lambda(::SplineSpec) = false
has_lambda(::SplineSpec{:lrs}) = true


###################################################################################################
# Small broadcast helpers
#
# `D` is the bin dimension of the working arrays; the value being transformed is a singleton
# along it, so every gather below is a masked sum that broadcasts back into place.

# `1:n` laid out along dimension `D` of an `ND`-dimensional array. A reshaped range rather than
# a materialized array: it is isbits, so it broadcasts against a CuArray unchanged.
@inline _binaxis(::Val{ND}, ::Val{D}, n::Int) where {ND,D} =
    reshape(1:n, ntuple(i -> i == D ? n : 1, Val(ND)))

# softplus(βz)/β, and its derivative σ(βz) recovered from the value itself -- σ(u) = 1 -
# exp(-softplus(u)) -- so the backward pass never has to keep the raw parameter around.
@inline _softplus_β(z, β) = max(z, zero(z)) + log1p(exp(-abs(β*z)))/β
@inline _dsoftplus_β(sp, β) = -expm1(-β*sp)

# Slice along the bin dimension.
@inline _bins(A, ::Val{D}, r) where {D} = selectdim(A, D, r)

@inline _slab(A, ::Val{D}) where {D} = ntuple(i -> i == D ? 1 : size(A, i), Val(ndims(A)))


###################################################################################################
# Knots

"""
    kn = spline_knots(θ, spec, Val(D))

 Turn the conditioner's raw output `θ` -- a tensor whose dimension `D` has length
 `n_spline_params(spec)` -- into the knots of a monotonic spline: bin widths and heights
 (softmax, so they tile `[-B, B]` exactly), the cumulative knot positions, the knot derivatives
 (softplus, so they stay positive), and for `:lrs` the per-bin weight `λ`.

 Held as a struct so that `spline_forward`, `spline_inverse` and `spline_vjp` share one
 normalization pass.
"""
struct SplineKnots{A,L}
    binw::A      # bin widths in x, length K along D, summing to 2B
    binh::A      # bin heights in y
    left::A      # left x-knot of each bin
    bottom::A    # left y-knot of each bin
    deriv::A     # derivative at each of the K+1 knots
    lambda::L    # per-bin interpolation weight for `:lrs`, `nothing` otherwise
end

function spline_knots(θ::AbstractArray{T}, spec::SplineSpec, ::Val{D}) where {T,D}
    K    = spec.nbins
    twoB = 2*T(spec.bound)

    binw, left   = _knot_axis(_bins(θ, Val(D), 1:K), T(spec.min_bin_width), twoB, K, Val(D))
    binh, bottom = _knot_axis(_bins(θ, Val(D), K+1:2K), T(spec.min_bin_height), twoB, K, Val(D))

    off   = has_lambda(spec) ? 3K : 2K
    deriv = _knot_derivs(_bins(θ, Val(D), off+1:off+n_deriv_params(spec)), spec, Val(D))

    return SplineKnots(binw, binh, left, bottom, deriv, _knot_lambda(θ, spec, Val(D)))
end

# Bin extents and the running knot positions they imply. The softmax is what pins the bins to
# tile `[-B, B]` exactly, so the spline meets its tails without a seam.
function _knot_axis(raw, floor_, twoB::T, K::Int, ::Val{D}) where {T,D}
    p     = softmax(raw; dims=D)
    extent = twoB .* (floor_ .+ (one(T) - K*floor_) .* p)
    right = cumsum(extent; dims=D) .- twoB/2
    return extent, right .- extent
end

# Derivatives at the K+1 knots. `:rqs` and `:lrs` pin both ends to 1 to match the linear tails:
# leaving them free makes the density discontinuous at ±B, which in the authors' experience
# shows up as failed optimization rather than as a slightly worse fit.
function _knot_derivs(raw, spec::SplineSpec, ::Val{D}) where {D}
    T = eltype(raw)
    δ = T(spec.min_derivative) .+ _softplus_β.(raw, _softplus_beta(spec, T))
    return _pad_derivs(δ, spec, Val(D))
end

# With `identity_init`, softplus_β(0) = 1 - min_derivative, so all-zero parameters give unit
# derivatives at every knot -- together with the uniform bins a zero softmax gives, exactly the
# identity map.
@inline _softplus_beta(spec::SplineSpec, ::Type{T}) where {T} =
    spec.identity_init ? T(log(2)/(1 - spec.min_derivative)) : one(T)

function _pad_derivs(δ, ::Union{SplineSpec{:rqs},SplineSpec{:lrs}}, ::Val{D}) where {D}
    ends = fill!(similar(δ, _slab(δ, Val(D))), one(eltype(δ)))
    return cat(ends, δ, ends; dims=D)
end

# On a circle the two ends are the same knot, so `δ[1]` does double duty as δ⁽⁰⁾ and δ⁽ᴷ⁾. That
# shared value is what keeps the density continuous across the seam.
_pad_derivs(δ, ::SplineSpec{:circular}, ::Val{D}) where {D} =
    cat(δ, _bins(δ, Val(D), 1:1); dims=D)

# The stored derivatives, as they sit inside the padded K+1 array.
_internal_derivs(deriv, spec::Union{SplineSpec{:rqs},SplineSpec{:lrs}}, ::Val{D}) where {D} =
    _bins(deriv, Val(D), 2:spec.nbins)
_internal_derivs(deriv, spec::SplineSpec{:circular}, ::Val{D}) where {D} =
    _bins(deriv, Val(D), 1:spec.nbins)

_knot_lambda(::AbstractArray, ::SplineSpec, ::Val) = nothing
function _knot_lambda(θ::AbstractArray{T}, spec::SplineSpec{:lrs}, ::Val{D}) where {T,D}
    K = spec.nbins
    m = T(spec.min_lambda)
    return m .+ (one(T) - 2m) .* _sigmoid.(_bins(θ, Val(D), 2K+1:3K))
end


###################################################################################################
# Domain and bin lookup
#
# `:rqs` and `:lrs` are the identity outside `[-B, B]`, so an unbounded input is fine and the
# spline only ever has to be evaluated on the clamped value -- which also keeps the discarded
# branch finite instead of NaN. `:circular` has no outside: the input is wrapped into `[-B, B)`.

@inline function _domain(x::AbstractArray{T}, spec::SplineSpec) where {T}
    B = T(spec.bound)
    return clamp.(x, -B, B), (x .> -B) .& (x .< B)
end

@inline function _domain(x::AbstractArray{T}, spec::SplineSpec{:circular}) where {T}
    B = T(spec.bound)
    return mod.(x .+ B, 2B) .- B, true
end

# Which bin each value falls in, as an index along the bin axis.
@inline _bin_index(t, edges, K::Int, ::Val{D}) where {D} =
    clamp.(sum(t .>= edges; dims=D), 1, K)

# Stride of axis `m`, for the dense arrays `spline_knots` produces -- every field of
# `SplineKnots` is a freshly materialized broadcast, `cumsum` or `cat`, so a size product is
# the true stride.
@inline function _axis_stride(A::AbstractArray, m::Int)
    s = 1
    for i in 1:m-1
        s *= size(A, i)
    end
    return s
end

# One axis's contribution to a linear index into `field`, laid out along that axis.
#
# The bin axis contributes nothing here -- `_bin_linear` adds its term from `k` instead -- but it
# still returns a range rather than a plain `0`, so that the tuple of offsets is homogeneous.
# With a `Union{Int,ReshapedArray}` element type inference gives up and the return type of a
# spline forward pass stops being concrete, which the type-stability tests catch.
#
# Offsets are *strided* ranges rather than `range .* stride`: the latter materializes a host
# `Array`, which then has to be copied into any GPU broadcast it meets.
@inline function _axis_offset(field::AbstractArray, m::Int, ::Val{ND}, ::Val{D}) where {ND,D}
    # `0:1:0` is the single offset 0 -- a zero *step* is not a legal range, and the step is
    # irrelevant for a length-1 range.
    m == D && return reshape(0:1:0, ntuple(_ -> 1, Val(ND)))
    st, n = _axis_stride(field, m), size(field, m)
    return reshape(0:st:st*(n-1), ntuple(i -> i == m ? n : 1, Val(ND)))
end

# Linear indices that read `field` at bin `k`, one per element of `k`.
#
# This is the gather that `sum(field .* (axis .== k); dims=D)` was performing. That form
# multiplies and reduces the entire bin axis in order to keep one entry of it, so it does `K`
# times the arithmetic and allocates two `K`-wide temporaries on the way; at `K = 8` bins it was
# the largest single cost in a spline coupling layer.
#
# The offsets follow each of the *field's* own axes, so a field whose axis is singleton where
# the data's is not broadcasts across it exactly as it did under the masked sum. That is the
# case for [`SplineLayer`](@ref), whose knots are the layer's own parameters and carry neither
# the spatial extent nor the batch.
@inline function _bin_linear(field::AbstractArray, k, ::Val{ND}, ::Val{D}) where {ND,D}
    off = ntuple(m -> _axis_offset(field, m, Val(ND), Val(D)), Val(ND))
    return broadcast(+, 1 .+ (k .- 1) .* _axis_stride(field, D), off...)
end

# Everything the spline formulas need for the bin each value landed in.
#
# `masks` asks for the one-hot and the "strictly earlier bins" mask over the bin axis. Only the
# VJP scatters onto that axis and needs them; `spline_forward` and `spline_inverse` were paying
# for two `K`-wide boolean arrays they never read.
@inline function _active_bin(t, edges, kn::SplineKnots, spec::SplineSpec, ::Val{D},
                             ::Val{masks}) where {D,masks}
    K  = spec.nbins
    ND = ndims(edges)
    k  = _bin_index(t, edges, K, Val(D))

    # One index array serves every field with a `K`-long bin axis: they are all slices of the
    # same parameter tensor, so they all share a shape.
    ik = _bin_linear(kn.binw, k, Val(ND), Val(D))

    # The padded derivatives are `K+1` long and so need their own; bin `k` runs from entry `k`
    # to entry `k+1`, one bin-axis stride further along.
    id = _bin_linear(kn.deriv, k, Val(ND), Val(D))
    sd = _axis_stride(kn.deriv, D)

    return (hot = masks ? _binaxis(Val(ND), Val(D), K) .== k : nothing,
            lt  = masks ? _binaxis(Val(ND), Val(D), K) .< k : nothing,
            k = k,
            wk = kn.binw[ik], hk = kn.binh[ik],
            xk = kn.left[ik], yk = kn.bottom[ik],
            dl = kn.deriv[id], dr = kn.deriv[id .+ sd],
            λ = _gather_lambda(kn.lambda, ik))
end

@inline _gather_lambda(::Nothing, ik) = nothing
@inline _gather_lambda(λ, ik) = λ[ik]


###################################################################################################
# Rational-quadratic splines
#
# In the bin `[x⁽ᵏ⁾, x⁽ᵏ⁺¹⁾]`, with `s` the secant slope, `ξ` the position within the bin and
# `δˡ, δʳ` the derivatives at its ends (Durkan et al., eq. 4-5):
#
#   y  = y⁽ᵏ⁾ + h·[s ξ² + δˡ ξ(1-ξ)] / [s + (δʳ + δˡ - 2s) ξ(1-ξ)]
#   y' = s²·[δʳ ξ² + 2 s ξ(1-ξ) + δˡ (1-ξ)²] / [s + (δʳ + δˡ - 2s) ξ(1-ξ)]²

# Shared so that the value, the log-derivative and the VJP agree by construction.
@inline function _rqs_terms(ξ, wk, hk, dl, dr)
    T  = eltype(ξ)
    s  = hk ./ wk
    u  = ξ .* (one(T) .- ξ)
    A  = s .* ξ .* ξ .+ dl .* u
    Dd = s .+ (dr .+ dl .- 2 .* s) .* u
    Q  = dr .* ξ .* ξ .+ 2 .* s .* u .+ dl .* (one(T) .- ξ) .* (one(T) .- ξ)
    return s, u, A, Dd, Q
end

function _rqs_value(ξ, b)
    s, _, A, Dd, Q = _rqs_terms(ξ, b.wk, b.hk, b.dl, b.dr)
    return b.yk .+ b.hk .* A ./ Dd, 2 .* log.(s) .+ log.(Q) .- 2 .* log.(Dd)
end

# Inverting eq. 4 is a quadratic in ξ (eq. 6-8). The root is taken in the `2c/(-b-√Δ)` form,
# which stays accurate as `a` approaches zero -- the near-affine bins, which is most of them
# once a flow has trained.
function _rqs_root(r, b)
    T  = eltype(r)
    s  = b.hk ./ b.wk
    c2 = b.dr .+ b.dl .- 2 .* s
    qa = b.hk .* (s .- b.dl) .+ r .* c2
    qb = b.hk .* b.dl .- r .* c2
    qc = .-s .* r
    disc = max.(qb .* qb .- 4 .* qa .* qc, zero(T))
    return 2 .* qc ./ (.-qb .- sqrt.(disc))
end


###################################################################################################
# Linear rational splines
#
# Each bin carries an extra knot at `x⁽ᵐ⁾ = (1-λ)x⁽ᵏ⁾ + λx⁽ᵏ⁺¹⁾` and is fitted by two homographic
# pieces (Dolatabadi et al., eq. 5, via Fuhr and Kallay's Algorithm 1). Both pieces are Möbius
# maps, so the inverse is a linear solve rather than a quadratic root -- the reason to prefer
# this family when a flow is sampled from far more often than it is scored.

# The intermediate knot, with the free weight `w⁽ᵏ⁾` fixed to 1: it cancels out of everything
# below, so it is not a degree of freedom.
@inline function _lrs_terms(wk, hk, dl, dr, λ)
    T  = eltype(wk)
    s  = hk ./ wk
    w2 = sqrt.(dl ./ dr)
    Z  = (one(T) .- λ) .+ λ .* w2
    M  = λ .* w2 .* hk ./ Z                       # y⁽ᵐ⁾ - y⁽ᵏ⁾
    wm = (λ .* dl .+ (one(T) .- λ) .* w2 .* dr) ./ s
    return s, w2, Z, M, wm
end

function _lrs_value(φ, b)
    T = eltype(φ)
    _, w2, _, M, wm = _lrs_terms(b.wk, b.hk, b.dl, b.dr, b.λ)
    H = b.hk .- M

    # Each piece is evaluated on its own clamped coordinate, so the branch that gets discarded
    # is still finite: a `log` of a negative denominator would throw, not quietly give NaN.
    φ1  = min.(φ, b.λ)
    Dd1 = (b.λ .- φ1) .+ wm .* φ1
    y1  = b.yk .+ wm .* φ1 .* M ./ Dd1
    l1  = log.(b.λ) .+ log.(wm) .+ log.(M) .- log.(b.wk) .- 2 .* log.(Dd1)

    φ2  = max.(φ, b.λ)
    Dd2 = wm .* (one(T) .- φ2) .+ w2 .* (φ2 .- b.λ)
    y2  = b.yk .+ M .+ w2 .* (φ2 .- b.λ) .* H ./ Dd2
    l2  = log1p.(.-b.λ) .+ log.(wm) .+ log.(w2) .+ log.(H) .- log.(b.wk) .- 2 .* log.(Dd2)

    lower = φ .<= b.λ
    return ifelse.(lower, y1, y2), ifelse.(lower, l1, l2)
end

# Both pieces invert by solving a linear equation; `r = y - y⁽ᵏ⁾` against `M` picks which.
function _lrs_root(r, b)
    T = eltype(r)
    _, w2, _, M, wm = _lrs_terms(b.wk, b.hk, b.dl, b.dr, b.λ)

    r1 = min.(r, M)
    φ1 = b.λ .* r1 ./ (r1 .- wm .* (r1 .- M))

    r2 = max.(r, M)
    φ2 = b.λ .+ (one(T) .- b.λ) .* wm .* (r2 .- M) ./
                (wm .* (r2 .- M) .- w2 .* (r2 .- b.hk))

    return ifelse.(r .<= M, φ1, φ2)
end


###################################################################################################
# Forward / inverse

@inline _value(ξ, b, ::SplineSpec) = _rqs_value(ξ, b)
@inline _value(ξ, b, ::SplineSpec{:lrs}) = _lrs_value(ξ, b)
@inline _root(r, b, ::SplineSpec) = _rqs_root(r, b)
@inline _root(r, b, ::SplineSpec{:lrs}) = _lrs_root(r, b)

"""
    y, logderiv = spline_forward(x, kn, spec, Val(D))

 Apply the spline elementwise, returning the transformed values and `log|dy/dx|` at each of
 them. Both come back with a singleton bin dimension, as `x` had.
"""
function spline_forward(x::AbstractArray{T}, kn::SplineKnots, spec::SplineSpec,
                        ::Val{D}) where {T,D}
    xc, inside = _domain(x, spec)
    b = _active_bin(xc, kn.left, kn, spec, Val(D), Val(false))
    y, lg = _value((xc .- b.xk) ./ b.wk, b, spec)
    return ifelse.(inside, y, x), ifelse.(inside, lg, zero(T))
end

"""
    x, logderiv = spline_inverse(y, kn, spec, Val(D))

 Invert the spline elementwise. `logderiv` is `log|dy/dx|` of the *forward* map evaluated at the
 recovered `x`, matching the convention of the layers, which negate it for the inverse
 direction.
"""
function spline_inverse(y::AbstractArray{T}, kn::SplineKnots, spec::SplineSpec,
                        ::Val{D}) where {T,D}
    yc, inside = _domain(y, spec)
    b = _active_bin(yc, kn.bottom, kn, spec, Val(D), Val(false))
    ξ = clamp.(_root(yc .- b.yk, b, spec), zero(T), one(T))
    _, lg = _value(ξ, b, spec)
    return ifelse.(inside, b.xk .+ ξ .* b.wk, y), ifelse.(inside, lg, zero(T))
end


###################################################################################################
# Vector-Jacobian product
#
# The chain runs in two stages. First the map from the active bin's geometry
# `(x⁽ᵏ⁾, w, y⁽ᵏ⁾, h, δˡ, δʳ, λ)` to `(y, log y')`, which is where the two spline families
# differ. Then the shared normalization: the per-bin residuals scatter onto the width and height
# vectors -- each knot is a running total, so bin `j` feels every later bin's knot residual --
# and go back through the softmax and softplus.

"""
    Δx, Δθ = spline_vjp(Δy, Δl, x, kn, spec, Val(D))

 Residual of the spline transform with respect to its input and its raw parameters, evaluated at
 the forward-direction input `x`. `Δl` is the residual on the elementwise `log|dy/dx|` and may
 be a scalar; pass `zero(T)` when the layer carries no log-determinant.
"""
function spline_vjp(Δy::AbstractArray{T}, Δl, x::AbstractArray{T}, kn::SplineKnots,
                    spec::SplineSpec, ::Val{D}) where {T,D}
    K = spec.nbins
    xc, inside = _domain(x, spec)
    b  = _active_bin(xc, kn.left, kn, spec, Val(D), Val(true))
    gx, gw, gh, gdl, gdr, gλ = _bin_vjp(Δy, Δl, (xc .- b.xk) ./ b.wk, b, spec)

    # Outside the spline interval the map is the identity: the input passes its residual through
    # untouched and the parameters see nothing at all. The residual on the bin's left knot is
    # the input's with the sign flipped -- moving the knot right is moving the input left.
    gxk = .-gx .* inside
    gyk = Δy .* inside
    Δw  = _axis_vjp(gw .* inside .* b.hot .+ gxk .* b.lt, kn.binw,
                    T(spec.min_bin_width), 2*T(spec.bound), K, Val(D))
    Δh  = _axis_vjp(gh .* inside .* b.hot .+ gyk .* b.lt, kn.binh,
                    T(spec.min_bin_height), 2*T(spec.bound), K, Val(D))
    Δd  = _deriv_vjp(gdl .* inside, gdr .* inside, b, kn, spec, Val(D))
    Δλ  = _lambda_vjp(gλ, inside, b, kn, spec, Val(D))

    return ifelse.(inside, gx, Δy), _cat_params(Δw, Δh, Δd, Δλ, Val(D))
end

## Rational-quadratic, and circular, which differs only in how δ is assembled
function _bin_vjp(Δy, Δl, ξ, b, ::SplineSpec)
    T = eltype(ξ)
    s, u, A, Dd, Q = _rqs_terms(ξ, b.wk, b.hk, b.dl, b.dr)
    v   = one(T) .- ξ
    Dd2 = Dd .* Dd
    g   = s .* s .* Q ./ Dd2                       # dy/dx

    # ∂y/∂·, holding the other bin quantities fixed.
    R   = A ./ Dd
    ys  = b.hk .* (ξ .* ξ .* Dd .- A .* (one(T) .- 2 .* u)) ./ Dd2
    ydl = b.hk .* u .* (Dd .- A) ./ Dd2
    ydr = .-b.hk .* u .* A ./ Dd2

    # ∂log(dy/dx)/∂·.
    Lξ  = (2 .* b.dr .* ξ .+ 2 .* s .* (one(T) .- 2 .* ξ) .- 2 .* b.dl .* v) ./ Q .-
          2 .* ((b.dr .+ b.dl .- 2 .* s) .* (one(T) .- 2 .* ξ)) ./ Dd
    Ls  = 2 ./ s .+ 2 .* u ./ Q .- 2 .* (one(T) .- 2 .* u) ./ Dd
    Ldl = v .* v ./ Q .- 2 .* u ./ Dd
    Ldr = ξ .* ξ ./ Q .- 2 .* u ./ Dd

    gx = Δy .* g .+ Δl .* Lξ ./ b.wk               # ξ and x differ by the fixed 1/w
    Gs = Δy .* ys .+ Δl .* Ls                      # w and h both act through s
    gh = Δy .* R .+ Gs ./ b.wk
    gw = .-ξ .* gx .- s .* Gs ./ b.wk
    return gx, gw, gh, Δy .* ydl .+ Δl .* Ldl, Δy .* ydr .+ Δl .* Ldr, nothing
end

## Linear rational
function _bin_vjp(Δy, Δl, φ, b, ::SplineSpec{:lrs})
    T = eltype(φ)
    λ = b.λ
    s, w2, Z, M, wm = _lrs_terms(b.wk, b.hk, b.dl, b.dr, λ)
    H = b.hk .- M
    lower = φ .<= λ

    # Piece below the intermediate knot: y = y⁽ᵏ⁾ + wm φ M / [(λ-φ) + wm φ].
    φ1  = min.(φ, λ)
    Dd1 = (λ .- φ1) .+ wm .* φ1
    q1  = Dd1 .* Dd1
    a_φ1, a_wm1 = λ .* wm .* M ./ q1, φ1 .* M .* (λ .- φ1) ./ q1
    a_M1, a_λ1  = wm .* φ1 ./ Dd1, .-wm .* φ1 .* M ./ q1
    l_φ1, l_wm1 = .-2 .* (wm .- one(T)) ./ Dd1, one(T) ./ wm .- 2 .* φ1 ./ Dd1
    l_λ1        = one(T) ./ λ .- 2 ./ Dd1

    # Piece above it: y = y⁽ᵏ⁾ + M + w2 (φ-λ) H / [wm(1-φ) + w2(φ-λ)].
    φ2  = max.(φ, λ)
    e, c = φ2 .- λ, one(T) .- φ2
    Dd2 = wm .* c .+ w2 .* e
    q2  = Dd2 .* Dd2
    a_φ2, a_wm2 = w2 .* wm .* H .* (one(T) .- λ) ./ q2, .-w2 .* e .* H .* c ./ q2
    a_w22, a_H2 = e .* H .* wm .* c ./ q2, w2 .* e ./ Dd2
    a_λ2        = .-w2 .* H .* wm .* c ./ q2
    l_φ2, l_wm2 = .-2 .* (w2 .- wm) ./ Dd2, one(T) ./ wm .- 2 .* c ./ Dd2
    l_w22, l_λ2 = one(T) ./ w2 .- 2 .* e ./ Dd2, 2 .* w2 ./ Dd2 .- one(T) ./ (one(T) .- λ)

    Gφ  = Δy .* ifelse.(lower, a_φ1, a_φ2)  .+ Δl .* ifelse.(lower, l_φ1, l_φ2)
    Gwm = Δy .* ifelse.(lower, a_wm1, a_wm2) .+ Δl .* ifelse.(lower, l_wm1, l_wm2)
    Gλd = Δy .* ifelse.(lower, a_λ1, a_λ2)  .+ Δl .* ifelse.(lower, l_λ1, l_λ2)
    Gw2 = ifelse.(lower, zero(T), Δy .* a_w22 .+ Δl .* l_w22)

    # `H = h - M`, so the upper piece's height residual splits between the two.
    GH  = ifelse.(lower, zero(T), Δy .* a_H2 .+ Δl ./ H)
    GM  = ifelse.(lower, Δy .* a_M1 .+ Δl ./ M, Δy) .- GH

    # Through the intermediate knot and the weights that place it.
    Gw2 = Gw2 .+ GM .* (λ .* (one(T) .- λ) .* b.hk ./ (Z .* Z)) .+
                 Gwm .* ((one(T) .- λ) .* b.dr ./ s)
    Gλ  = Gλd .+ GM .* (w2 .* b.hk ./ (Z .* Z)) .+ Gwm .* ((b.dl .- w2 .* b.dr) ./ s)
    Gs  = .-Gwm .* wm ./ s

    gx  = Gφ ./ b.wk
    gh  = GH .+ GM .* (M ./ b.hk) .+ Gs ./ b.wk
    gw  = .-Δl ./ b.wk .- φ .* gx .- s .* Gs ./ b.wk    # log y' carries an explicit -log w
    gdl = Gw2 .* (w2 ./ (2 .* b.dl)) .+ Gwm .* (λ ./ s)
    gdr = .-Gw2 .* (w2 ./ (2 .* b.dr)) .+ Gwm .* ((one(T) .- λ) .* w2 ./ s)

    return gx, gw, gh, gdl, gdr, Gλ
end

# Softmax VJP. The probabilities are read back off the knots rather than recomputed: they are an
# affine function of the bin extents the layer already holds.
function _axis_vjp(ΔS, extent, floor_, twoB::T, K::Int, ::Val{D}) where {T,D}
    c = one(T) - K*floor_
    p = (extent ./ twoB .- floor_) ./ c
    t = (twoB * c) .* ΔS
    return p .* (t .- sum(p .* t; dims=D))
end

# Scatter the two end-derivative residuals onto the K+1 knots, drop the pinned boundaries (or
# fold the shared one back onto itself, for a circular spline), then go back through softplus.
function _deriv_vjp(gdl, gdr, b, kn::SplineKnots, spec::SplineSpec, ::Val{D}) where {D}
    T  = eltype(kn.deriv)
    K  = spec.nbins
    ax = _binaxis(Val(ndims(kn.deriv)), Val(D), K+1)
    # The bin index comes along with the gathered bin, so it needs no recovering from the mask.
    Δδ = gdl .* (ax .== b.k) .+ gdr .* (ax .== b.k .+ 1)
    δ  = _internal_derivs(kn.deriv, spec, Val(D))
    return _unpad_derivs(Δδ, spec, Val(D)) .*
           _dsoftplus_β.(δ .- T(spec.min_derivative), _softplus_beta(spec, T))
end

_unpad_derivs(Δδ, spec::Union{SplineSpec{:rqs},SplineSpec{:lrs}}, ::Val{D}) where {D} =
    _bins(Δδ, Val(D), 2:spec.nbins)

function _unpad_derivs(Δδ, spec::SplineSpec{:circular}, ::Val{D}) where {D}
    K = spec.nbins
    g = copy(_bins(Δδ, Val(D), 1:K))
    _bins(g, Val(D), 1:1) .+= _bins(Δδ, Val(D), K+1:K+1)
    return g
end

_lambda_vjp(::Nothing, inside, b, kn, ::SplineSpec, ::Val) = nothing
function _lambda_vjp(gλ, inside, b, kn::SplineKnots, spec::SplineSpec{:lrs}, ::Val{D}) where {D}
    T = eltype(kn.lambda)
    m = T(spec.min_lambda)
    # λ = min_λ + (1-2min_λ)·σ(z), so the sigmoid's derivative comes off the value itself. Only
    # the active bin's λ took part, hence the one-hot.
    u = (kn.lambda .- m) ./ (one(T) - 2m)
    return (gλ .* inside) .* b.hot .* ((one(T) - 2m) .* u .* (one(T) .- u))
end

_cat_params(Δw, Δh, Δd, ::Nothing, ::Val{D}) where {D} = cat(Δw, Δh, Δd; dims=D)
_cat_params(Δw, Δh, Δd, Δλ, ::Val{D}) where {D} = cat(Δw, Δh, Δλ, Δd; dims=D)
