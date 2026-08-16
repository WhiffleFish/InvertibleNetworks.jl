# A flow on a bounded domain, with a uniform base distribution.
#
# Change of variables does not care which base a flow pushes forward: for any `p_z` and
# bijection `f`, `log p_X(x) = log p_z(f(x)) + log|det J_f(x)|`. Gaussian is a convention
# here, not a constraint, and `base` is the seam for changing it.
#
# The construction below is the one that motivates the feature. Modelling a density on a
# bounded domain is usually done with a flow on all of Rᵈ squashed into the box by a sigmoid,
# which costs: a uniform is not representable (the pullback of a uniform through a sigmoid is
# logistic, not Gaussian, so the flow has to learn a Gaussian→logistic map just to sit still),
# normalization is only approximate, and `sigmoid(u)` rounds to exactly 1 in Float32 once
# `u ≳ 16.6`, sending `log(y(1-y))` to -Inf.
#
# A rational-quadratic spline with linear tails is already a bijection of `[-B, B]` onto
# itself, so a stack of `CouplingLayerSpline(...; mix=false)` is a bijection of `[-B, B]ᵈ`.
# Pair it with `BoxUniform(B)` and none of the three costs appear.

using InvertibleNetworks, Flux, LinearAlgebra, Random, Test

Random.seed!(0)

k, n_hidden, batchsize = 4, 32, 16
B = 3f0
base = BoxUniform(B)

# Uniform data on the box, and a chain that is a bijection of it. `mix=false` matters: the
# `Conv1x1` a coupling layer puts in front by default is a rotation, which takes points out of
# the box. Alternating `swap` is how every channel still gets transformed.
X = rand(base, 1, 1, k, batchsize)
flow = InvertibleChain(
    CouplingLayerSpline(k, n_hidden; bound=B, mix=false, logdet=true),
    CouplingLayerSpline(k, n_hidden; bound=B, mix=false, swap=true, logdet=true),
    SplineLayer(k; bound=B, logdet=true))

@test preserves_box(flow, B)
@test isnothing(check_latent_support(flow, base))

# At the identity initialization the two constants of the change of variables cancel and the
# model is *exactly* uniform -- not approximately, and at Float32 precision.
@test isapprox(log_likelihood(X, flow; base=base, normalized=true),
               Float32(-k*log(2B)); atol=1f-5)

# The invariant is structural, not a property of the initialization: however far the
# parameters move, the flow maps the box onto itself in both directions.
for p in get_params(flow)
    p.data .+= 0.5f0 .* randn(Float32, size(p.data))
end
Z = flow.forward(X)[1]
@test all(abs.(Z) .<= B)
@test all(abs.(flow.inverse(rand(base, 1, 1, k, batchsize))) .<= B)

# So the density integrates to 1 over the box at any parameter value, with no tail escaping
# the domain of integration -- checked here by quadrature in d = 2.
flow2 = InvertibleChain(CouplingLayerSpline(2, 8; bound=B, mix=false, logdet=true),
                        CouplingLayerSpline(2, 8; bound=B, mix=false, swap=true, logdet=true))
for p in get_params(flow2)
    p.data .+= 0.3f0 .* randn(Float32, size(p.data))
end
ngrid = 300
h = 2*Float64(B)/ngrid
grid = collect(range(-Float64(B) + h/2, Float64(B) - h/2; length=ngrid))
pts = reshape(reduce(hcat, [Float32[x1, x2] for x1 in grid for x2 in grid]), 1, 1, 2, ngrid^2)
mass = sum(exp.(Float64.(log_likelihood_per_sample(pts, flow2; base=base, normalized=true))))*h^2
@test isapprox(mass, 1.0; atol=1e-3)

# Training is the usual objective. The uniform is flat on its support, so the gradient comes
# entirely from the log-determinant.
opt_state = Flux.setup(Adam(1f-3), flow)
for i = 1:10
    l, grads = Flux.withgradient(m -> -log_likelihood(X, m; base=base), flow)
    Flux.update!(opt_state, flow, grads[1])
end
@test isfinite(log_likelihood(X, flow; base=base, normalized=true))

# Sampling and scoring in one pass, with `rand` drawing from the base rather than a hard-coded
# `randn` -- the second place the Gaussian assumption otherwise hides.
Xs, logp = inverse_and_log_likelihood_per_sample(rand(base, 1, 1, k, batchsize), flow;
                                                 base=base, normalized=true)
@test all(abs.(Xs) .<= B)
@test length(logp) == batchsize

# Pairing a bounded base with a network that is not a bijection of the box is a modelling
# error, and is reported as one instead of returning a quietly wrong density.
glow = InvertibleChain(ActNorm(k; logdet=true), Conv1x1(k),
                       CouplingLayerGlow(k, n_hidden; logdet=true))
glow(X)
@test_throws ArgumentError log_likelihood(X, glow; base=base)
println(try; check_latent_support(glow, base); catch e; e.msg; end)

# The bounds also have to nest the right way round: with a base narrower than the spline's own
# interval, a point inside the base's box can be mapped out of it.
@test_throws ArgumentError check_latent_support(flow, BoxUniform(B - 1f0))

# Nothing above changes the default. `base` is omitted or `StandardNormal`, and the Gaussian
# path is exactly what it always was.
@test log_likelihood(X, glow) == log_likelihood(X, glow; base=StandardNormal())
@test log_likelihood(X, glow; σ=2f0) == log_likelihood(X, glow; base=StandardNormal(0f0, 2f0))
