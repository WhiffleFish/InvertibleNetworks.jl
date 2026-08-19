# Neural spline flow layers: coupling and elementwise.
#
# Durkan et al. (2019), https://arxiv.org/abs/1906.04032
#
# A spline coupling layer is a `CouplingLayerGlow` with the affine `Sm .* X1 .+ Tm` replaced by
# a monotonic spline whose knots the residual block predicts. Everything else -- the 1x1
# convolution, the half-and-half split, the recompute-from-the-output backward pass -- is the
# same, and so is the cost: one conditioner evaluation in each direction.
#
# The spline maths lives in `utils/splines.jl`; this file is the plumbing that reshapes the
# conditioner's output into knots and threads the residual back through it.

export CouplingLayerSpline, CouplingLayerSpline3D, SplineLayer


###################################################################################################
# Shared helpers
#
# The spline routines want a bin axis to broadcast over. Slotting it in right after the channel
# dimension means the conditioner's `(…, C₁·P, B)` output reshapes into `(…, C₁, P, B)` with no
# `permutedims`, and the value being transformed just gains a singleton there.

@inline _widen(X::AbstractArray{T,N}) where {T,N} =
    reshape(X, size(X)[1:N-1]..., 1, size(X, N))

@inline _narrow(Y::AbstractArray, sz::Tuple) = reshape(Y, sz)

# The residual on an elementwise `log|dy/dx|`, matching `apply_logdet_weight`: the layers
# accumulate the gradient of `-logdet`, and `logdet` is the batch average.
@inline _spline_logdet_adjoint(::Nothing, ::Type{T}, batch::Int, ::Val) where {T} =
    -one(T)/batch
@inline _spline_logdet_adjoint(w::AbstractVector, ::Type{T}, batch::Int, ::Val{ND}) where {T,ND} =
    reshape(w, ntuple(_ -> 1, Val(ND-1))..., :)

_spline_out(Y, lg, ::Val{false}) = Y
_spline_out(Y, lg, ::Val{true}) = (Y, sum(lg)/size(lg)[end])
_spline_out(Y, lg, ::Val{:sample}) = (Y, per_sample_sum(lg))

_spline_inv_out(X, lg, ::Val{false}) = X
_spline_inv_out(X, lg, ::Val{true}) = (X, -sum(lg)/size(lg)[end])
_spline_inv_out(X, lg, ::Val{:sample}) = (X, -per_sample_sum(lg))


###################################################################################################
# Coupling layer

"""
    CL = CouplingLayerSpline(n_in, n_hidden; spline=:rqs, nbins=8, bound=3f0, logdet=false,
                             ndims=2)

or

    CL = CouplingLayerSpline(C, RB, spec::SplineSpec; logdet=false, swap=false)

 Create a neural spline flow coupling layer: a 1x1 convolution and a half-and-half split, as in
 [`CouplingLayerGlow`](@ref), with the affine transform replaced by a monotonic spline whose
 knots the residual block predicts (Durkan et al., 2019).

 The point of the substitution is expressiveness per layer. An affine coupling can only shift
 and scale each channel, so a multi-modal conditional needs many layers to build up; a spline
 with `K` bins can bend the map differently in each bin, and the paper fits densities with
 hundreds of modes using two coupling layers. It costs one conditioner evaluation in each
 direction, exactly like the affine layer -- the inverse is still analytic and still one pass.

 *Input*:

 - `n_in`, `n_hidden`: number of input and hidden channels

 - `spline`: `:rqs`, `:circular` or `:lrs`; see [`SplineSpec`](@ref)

 - `nbins`, `bound`: bins `K` and half-interval `B` of the spline

 - `logdet`: bool to indicate whether to compute the logdet

 - `identity_init`: start the layer as the identity map, by scaling the derivative softplus and
   zeroing the conditioner's output layer. Recommended, and the reason a deep spline flow
   trains stably from step one; pass `zero_init=false` to keep the random conditioner

 - `mix`: put a [`Conv1x1`](@ref) in front, so that successive layers transform different
   directions. Defaults to `true`, except for `:circular` -- see below

 - `swap`: transform the *second* half of the channels, conditioned on the first, instead of the
   other way round. Alternating `swap` down a stack is how every channel gets transformed when
   `mix=false`

 - `k1`, `k2`, `p1`, `p2`, `s1`, `s2`, `ndims`, `nx`, `dense`, `freeze_conv`: as for
   [`CouplingLayerGlow`](@ref)

 *Circular splines* act on the torus `[-B, B)ᵈ`, and a rotation is not a map of the torus: a 1x1
 convolution would push the transformed channels off the domain and the layer would no longer
 invert. So `:circular` defaults to `mix=false` and rejects `mix=true`, following Rezende et al.
 (2020), who alternate which dimensions are transformed instead. Its input must already lie in
 `[-B, B)`; anything outside is wrapped into it, which is a projection and not invertible.

 *Output*:

 - `CL`: invertible spline coupling layer

 *Usage:*

 - Forward mode: `Y, logdet = CL.forward(X)`    (if constructed with `logdet=true`)

 - Inverse mode: `X = CL.inverse(Y)`

 - Backward mode: `ΔX, X = CL.backward(ΔY, Y)`

 - Per sample: pass `logdet=:sample` to either direction

 *Trainable parameters:*

 - None in `CL` itself; the parameters live in the residual block `CL.RB` and, when present, the
   1x1 convolution `CL.C`

 See also: [`SplineSpec`](@ref), [`SplineLayer`](@ref), [`CouplingLayerGlow`](@ref),
 [`Conv1x1`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct CouplingLayerSpline{C<:Union{Conv1x1,Nothing},R<:ConditionerBlock,S<:SplineSpec,LD} <: NeuralNetLayer
    C::C
    RB::R
    spline::S
    logdet::Bool
    swap::Bool
end

CouplingLayerSpline(C::CT, RB::RT, spec::ST, logdet::Bool, swap::Bool) where
    {CT<:Union{Conv1x1,Nothing},RT<:ConditionerBlock,ST<:SplineSpec} =
    CouplingLayerSpline{CT,RT,ST,logdet}(C, RB, spec, logdet, swap)

Flux.@layer CouplingLayerSpline

supports_per_sample_logdet(::CouplingLayerSpline) = true

Base.show(io::IO, L::CouplingLayerSpline) =
    print(io, "CouplingLayerSpline($(L.spline)$(isnothing(L.C) ? ", mix=false" : "")$(L.swap ? ", swap=true" : ""))")

# Constructor from an existing convolution, conditioner and spline shape.
CouplingLayerSpline(C::Union{Conv1x1,Nothing}, RB::ConditionerBlock,
                    spec::SplineSpec; logdet=false, swap=false) =
    CouplingLayerSpline(C, RB, spec, logdet, swap)

# Constructor from input dimensions
function CouplingLayerSpline(n_in::Int64, n_hidden::Int64; spline=:rqs, nbins=8, bound=3f0,
                             min_bin_width=1f-3, min_bin_height=1f-3, min_derivative=1f-3,
                             min_lambda=1f-3, identity_init::Bool=true, zero_init=nothing,
                             mix=nothing, swap::Bool=false,
                             nx=nothing, dense=false, freeze_conv=false,
                             k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, logdet=false, ndims=2)

    # A coupling layer splits its channels in two and conditions one half on the other, which
    # needs at least two channels.
    n_in < 2 && throw(ArgumentError(
        "CouplingLayerSpline needs at least 2 input channels to split, got n_in = $n_in"))

    spec = SplineSpec(spline; nbins=nbins, bound=bound, min_bin_width=min_bin_width,
                      min_bin_height=min_bin_height, min_derivative=min_derivative,
                      min_lambda=min_lambda, identity_init=identity_init)
    zi = isnothing(zero_init) ? identity_init : zero_init

    # A circular spline maps the torus to itself; a 1x1 convolution does not, so mixing would
    # move the transformed channels off the domain and cost the layer its inverse.
    circular = spline === :circular
    domix = isnothing(mix) ? !circular : mix
    circular && domix && throw(ArgumentError(
        "a circular spline acts on the torus [-B, B), which a 1x1 convolution does not " *
        "preserve, so `mix=true` would leave the layer non-invertible; use mix=false and " *
        "alternate `swap` between layers instead"))

    split_num = Int(round(n_in/2))
    n_trans   = swap ? n_in - split_num : split_num       # channels the spline transforms
    n_cond    = n_in - n_trans                            # channels the conditioner reads
    out_chan  = n_trans * n_spline_params(spec)

    if dense
        isnothing(nx) && error("Dense network needs nx as kwarg input")
        init = zi ? Flux.zeros32 : Flux.glorot_uniform
        RB = FluxBlock(Chain(x -> reshape(x, nx*n_cond, :),
                             Dense(nx*n_cond, n_in*n_hidden, relu),
                             Dense(n_in*n_hidden, n_in*n_hidden, relu),
                             Dense(n_in*n_hidden, nx*out_chan; init=init),
                             x -> reshape(x, nx, out_chan, :)))
    else
        # `fan=false` asks the residual block for a gated linear output rather than the
        # activation it applies when `fan=true`. That matters here: with the default ReLU the
        # conditioner could never emit a negative parameter, which would pin every knot
        # derivative at or above 1 and throw away most of what a spline is for. The gate halves
        # the channel count, hence the doubled `n_out`.
        RB = ResidualBlock(n_cond, n_hidden; n_out=2*out_chan, k1=k1, k2=k2, p1=p1, p2=p2,
                           s1=s1, s2=s2, fan=false, ndims=ndims)
        zi && fill!(RB.W3.data, 0)
    end

    return CouplingLayerSpline(domix ? Conv1x1(n_in; freeze=freeze_conv) : nothing,
                               RB, spec, logdet, swap)
end

CouplingLayerSpline3D(args...; kw...) = CouplingLayerSpline(args...; kw..., ndims=3)

# Conditioner output, reshaped so that the `P` spline parameters of each transformed channel sit
# on their own axis.
@inline _spline_params(flat::AbstractArray, Xt::AbstractArray{T,N}, spec::SplineSpec) where {T,N} =
    reshape(flat, size(Xt)[1:N-1]..., n_spline_params(spec), size(Xt, N))

# Optional 1x1 mixing.
@inline _mix_forward(X, ::Nothing) = X
@inline _mix_forward(X, C::Conv1x1) = conv1x1_forward(X, C)
@inline _mix_inverse(Y, ::Nothing) = Y
@inline _mix_inverse(Y, C::Conv1x1) = conv1x1_inverse(Y, C)
@inline _mix_grad(ΔX, X, ::Nothing; set_grad::Bool=true) =
    set_grad ? ΔX : (ΔX, Parameter[])
@inline _mix_grad(ΔX, X, C::Conv1x1; set_grad::Bool=true) =
    inverse_grad(ΔX, X, C; set_grad=set_grad)

# Which half the spline transforms, and how the two halves go back together.
@inline function _couple_split(X, swap::Bool)
    A, B = tensor_split(X)
    return swap ? (B, A) : (A, B)
end
@inline _couple_cat(Yt, Xc, swap::Bool) = swap ? tensor_cat(Xc, Yt) : tensor_cat(Yt, Xc)

# Forward pass: Input X, Output Y
forward(X::AbstractArray{T,N}, L::CouplingLayerSpline{C,R,S,LD}; logdet=nothing) where {T,N,C,R,S,LD} =
    _forward(X, L, logdet_mode(logdet, Val(LD)))

function _forward(X::AbstractArray{T,N}, L::CouplingLayerSpline, mode::Val) where {T,N}
    Xt, Xc = _couple_split(_mix_forward(X, L.C), L.swap)

    kn = spline_knots(_spline_params(block_forward(Xc, L.RB), Xt, L.spline), L.spline, Val(N))
    y, lg = spline_forward(_widen(Xt), kn, L.spline, Val(N))

    # The conditioning half passes through unchanged, and `tensor_cat` copies it anyway.
    return _spline_out(_couple_cat(_narrow(y, size(Xt)), Xc, L.swap), lg, mode)
end

# Inverse pass: Input Y, Output X
#
# The conditioner reads the untransformed half, so it gives the same knots in both directions and
# the spline inverts in a single pass. `logdet` defaults to `false` rather than to `L.logdet` so
# that the internal call in `backward` keeps its return shape.
function inverse(Y::AbstractArray{T,N}, L::CouplingLayerSpline; logdet=false) where {T,N}
    Yt, Xc = _couple_split(Y, L.swap)
    kn = spline_knots(_spline_params(block_forward(Xc, L.RB), Yt, L.spline), L.spline, Val(N))
    x, lg = spline_inverse(_widen(Yt), kn, L.spline, Val(N))
    X = _mix_inverse(_couple_cat(_narrow(x, size(Yt)), Xc, L.swap), L.C)
    return _spline_inv_out(X, lg, logdet_mode(logdet))
end

# Backward pass: Input (ΔY, Y), Output (ΔX, X)
function backward(ΔY::AbstractArray{T,N}, Y::AbstractArray{T,N}, L::CouplingLayerSpline;
                  set_grad::Bool=true, logdet_weight=nothing) where {T,N}
    check_logdet_weight(logdet_weight, set_grad)

    # Recompute the forward state, keeping the residual block's own intermediates: its backward
    # pass below would otherwise run its forward a second time.
    Yt, Xc = _couple_split(Y, L.swap)
    rb = block_forward_save(Xc, L.RB)
    kn = spline_knots(_spline_params(block_output(rb), Yt, L.spline), L.spline, Val(N))

    x, _ = spline_inverse(_widen(Yt), kn, L.spline, Val(N))
    Xt = _narrow(x, size(Yt))
    X  = _mix_inverse(_couple_cat(Xt, Xc, L.swap), L.C)

    ΔYt, ΔYc = _couple_split(ΔY, L.swap)
    Δl = L.logdet ? _spline_logdet_adjoint(logdet_weight, T, size(Y, N), Val(N+1)) : zero(T)
    Δx, Δθ = spline_vjp(_widen(ΔYt), Δl, x, kn, L.spline, Val(N))

    ΔXt = _narrow(Δx, size(Yt))
    Δθf = _narrow(Δθ, size(block_output(rb)))

    if set_grad
        ΔXc = block_backward(Δθf, Xc, rb, L.RB) .+ ΔYc
        return _mix_grad(_couple_cat(ΔXt, ΔXc, L.swap), X, L.C), X
    end

    ΔXc, Δθrb = block_backward(Δθf, Xc, rb, L.RB; set_grad=false)
    ΔXc = ΔXc .+ ΔYc
    ΔX, Δθc = _mix_grad(_couple_cat(ΔXt, ΔXc, L.swap), X, L.C; set_grad=false)
    Δθ_all = cat(Δθc, Δθrb; dims=1)
    L.logdet || return ΔX, Δθ_all, X

    # The Jacobian interface reports the log-determinant's own parameter gradient separately,
    # unweighted and with the sign of `+logdet`. Both passes reuse `rb`, so the conditioner's
    # forward pass is paid for once.
    _, Δθl = spline_vjp(zero(T), one(T)/size(Y, N), x, kn, L.spline, Val(N))
    _, ∇logdet = block_backward(_narrow(Δθl, size(block_output(rb))), Xc, rb, L.RB;
                                set_grad=false)
    return ΔX, Δθ_all, X, cat(0 .* Δθc, ∇logdet; dims=1)
end

adjointJacobian(ΔY::AbstractArray{T,N}, Y::AbstractArray{T,N}, L::CouplingLayerSpline) where {T,N} =
    backward(ΔY, Y, L; set_grad=false)


###################################################################################################
# Elementwise layer

"""
    SL = SplineLayer(k; spline=:rqs, nbins=8, bound=3f0, logdet=false)

 Create an elementwise monotonic spline layer over `k` channels: one spline per channel, with
 free trainable knots rather than knots predicted from other channels.

 This is the transform of eqs. 9-11 of Durkan et al. (2019), which the paper applies to the half
 of a coupling layer that would otherwise pass through untouched. It is also useful on its own
 as a learned, invertible, per-channel warp -- a strictly more flexible [`ActNorm`](@ref) --
 and, with `spline=:circular`, as the natural map for angle-valued channels.

 Being elementwise, it has no mixing of its own: put it in an [`InvertibleChain`](@ref) between
 layers that do mix, such as [`Conv1x1`](@ref) or a coupling layer.

 *Input*:

 - `k`: number of channels

 - `spline`, `nbins`, `bound`, `identity_init`: see [`SplineSpec`](@ref)

 - `logdet`: bool to indicate whether to compute the logdet

 *Usage:*

 - Forward mode: `Y, logdet = SL.forward(X)`    (if constructed with `logdet=true`)

 - Inverse mode: `X = SL.inverse(Y)`

 - Backward mode: `ΔX, X = SL.backward(ΔY, Y)`

 *Trainable parameters:*

 - The raw spline parameters `SL.θ`, of size `(k, n_spline_params(SL.spline))`. Initialized to
   zero, which under `identity_init` is exactly the identity map.

 See also: [`SplineSpec`](@ref), [`CouplingLayerSpline`](@ref), [`ActNorm`](@ref)
"""
struct SplineLayer{P<:Parameter,S<:SplineSpec,LD} <: NeuralNetLayer
    k::Int
    θ::P
    spline::S
    logdet::Bool
end

SplineLayer(k::Int, θ::PT, spec::ST, logdet::Bool) where {PT<:Parameter,ST<:SplineSpec} =
    SplineLayer{PT,ST,logdet}(k, θ, spec, logdet)

Flux.@layer SplineLayer

supports_per_sample_logdet(::SplineLayer) = true

Base.show(io::IO, L::SplineLayer) = print(io, "SplineLayer($(L.k), $(L.spline))")

function SplineLayer(k::Int; spline=:rqs, nbins=8, bound=3f0, min_bin_width=1f-3,
                     min_bin_height=1f-3, min_derivative=1f-3, min_lambda=1f-3,
                     identity_init::Bool=true, logdet=false)
    spec = SplineSpec(spline; nbins=nbins, bound=bound, min_bin_width=min_bin_width,
                      min_bin_height=min_bin_height, min_derivative=min_derivative,
                      min_lambda=min_lambda, identity_init=identity_init)
    return SplineLayer(k, Parameter(zeros(Float32, k, n_spline_params(spec))), spec, logdet)
end

# The per-channel parameters, broadcast over the spatial dimensions and the batch.
@inline _layer_params(L::SplineLayer, ::Val{N}) where {N} =
    reshape(L.θ.data, ntuple(_ -> 1, Val(N-2))..., size(L.θ.data)..., 1)

# Dimensions to sum the residual over to get back to the parameter's own shape: everything
# except the channel axis and the parameter axis it carries.
@inline _param_reduce_dims(::Val{N}) where {N} = ntuple(i -> i <= N-2 ? i : N+1, Val(N-1))

forward(X::AbstractArray{T,N}, L::SplineLayer{P,S,LD}; logdet=nothing) where {T,N,P,S,LD} =
    _forward(X, L, logdet_mode(logdet, Val(LD)))

function _forward(X::AbstractArray{T,N}, L::SplineLayer, mode::Val) where {T,N}
    kn = spline_knots(_layer_params(L, Val(N)), L.spline, Val(N))
    y, lg = spline_forward(_widen(X), kn, L.spline, Val(N))
    return _spline_out(_narrow(y, size(X)), lg, mode)
end

function inverse(Y::AbstractArray{T,N}, L::SplineLayer; logdet=false) where {T,N}
    kn = spline_knots(_layer_params(L, Val(N)), L.spline, Val(N))
    x, lg = spline_inverse(_widen(Y), kn, L.spline, Val(N))
    return _spline_inv_out(_narrow(x, size(Y)), lg, logdet_mode(logdet))
end

function backward(ΔY::AbstractArray{T,N}, Y::AbstractArray{T,N}, L::SplineLayer;
                  set_grad::Bool=true, logdet_weight=nothing) where {T,N}
    check_logdet_weight(logdet_weight, set_grad)

    kn = spline_knots(_layer_params(L, Val(N)), L.spline, Val(N))
    x, _ = spline_inverse(_widen(Y), kn, L.spline, Val(N))
    X = _narrow(x, size(Y))

    Δl = L.logdet ? _spline_logdet_adjoint(logdet_weight, T, size(Y, N), Val(N+1)) : zero(T)
    Δx, Δθ = spline_vjp(_widen(ΔY), Δl, x, kn, L.spline, Val(N))
    ΔX = _narrow(Δx, size(Y))
    Δθp = _narrow(sum(Δθ; dims=_param_reduce_dims(Val(N))), size(L.θ.data))

    if set_grad
        L.θ.grad = Δθp
        return ΔX, X
    end

    L.logdet || return ΔX, [Parameter(Δθp)], X
    _, Δθl = spline_vjp(zero(T), one(T)/size(Y, N), x, kn, L.spline, Val(N))
    ∇logdet = _narrow(sum(Δθl; dims=_param_reduce_dims(Val(N))), size(L.θ.data))
    return ΔX, [Parameter(Δθp)], X, [Parameter(∇logdet)]
end

adjointJacobian(ΔY::AbstractArray{T,N}, Y::AbstractArray{T,N}, L::SplineLayer) where {T,N} =
    backward(ΔY, Y, L; set_grad=false)
