# Pluggable base (latent) distributions and the domain invariant they carry.

using InvertibleNetworks, Flux, Test, Random, LinearAlgebra

Random.seed!(11)

const B = 3f0
const nb = 8

# A chain that is a bijection of [-B, B]ᵈ: spline couplings with no mixing, alternating which
# half they transform, plus an elementwise spline.
box_flow(; bound=B, k=4, nh=16) =
    InvertibleChain(CouplingLayerSpline(k, nh; bound=bound, mix=false, logdet=true),
                    CouplingLayerSpline(k, nh; bound=bound, mix=false, swap=true, logdet=true),
                    SplineLayer(k; bound=bound, logdet=true))

# The usual Glow-style chain, which is not.
free_flow(; k=4, nh=16) =
    InvertibleChain(ActNorm(k; logdet=true), Conv1x1(k), CouplingLayerGlow(k, nh; logdet=true))

inbox(k, n; bound=B) = (2*bound) .* rand(Float32, 1, 1, k, n) .- bound

# Push the parameters away from the identity initialization, so a test is not passing only
# because every layer happens to be the identity.
function perturb!(net, scale=0.4f0)
    for p in get_params(net)
        p.data .+= scale .* randn(Float32, size(p.data))
    end
    return net
end


@testset "distribution interface" begin
    X = randn(Float32, 1, 1, 4, nb)

    d = StandardNormal()
    @test d isa LatentDistribution
    @test latent_support(d) isa UnboundedSupport
    @test eltype(d) == Float32
    @test logpdf_per_sample(d, X) isa Vector{Float32}
    @test length(logpdf_per_sample(d, X)) == nb
    @test sum(logpdf_per_sample(d, X))/nb ≈ logpdf_mean(d, X)
    @test sum(logpdf_per_sample(d, X; normalized=true))/nb ≈
          logpdf_mean(d, X; normalized=true)

    u = BoxUniform(B)
    @test latent_support(u) == BoxSupport(B)
    @test sum(logpdf_per_sample(u, X))/nb ≈ logpdf_mean(u, X)

    # The uniform's log-density is nothing but its normalizing constant, so dropping the
    # constant leaves a flat zero -- and the constant is -d log 2B.
    Z = inbox(4, nb)
    @test all(iszero, logpdf_per_sample(u, Z))
    @test logpdf_per_sample(u, Z; normalized=true) ≈ fill(-4*log(2B), nb)
    @test uniform_lognorm(Z, B) ≈ Float32(-4*log(2B))

    # Outside the box the density is zero, i.e. the log-density is -Inf, not a finite number
    Zout = copy(Z); Zout[1, 1, 1, 1] = B + 1f0
    lp = logpdf_per_sample(u, Zout; normalized=true)
    @test lp[1] == -Inf32
    @test all(isfinite, lp[2:end])

    # Constructors validate
    @test_throws ArgumentError StandardNormal(0f0, -1f0)
    @test_throws ArgumentError BoxUniform(0f0)
    @test StandardNormal(0, 1) isa StandardNormal{Int}
    @test StandardNormal(0f0) == StandardNormal(0f0, 1f0)
end

@testset "sampling through the base" begin
    Random.seed!(3)
    u = BoxUniform(B)
    Zu = rand(u, 1, 1, 4, 512)
    @test eltype(Zu) == Float32
    @test size(Zu) == (1, 1, 4, 512)
    @test all(abs.(Zu) .<= B)
    @test isapprox(sum(Zu)/length(Zu), 0f0; atol=0.2f0)

    d = StandardNormal(1f0, 2f0)
    Zn = rand(d, 4, 4096)
    @test size(Zn) == (4, 4096)
    @test isapprox(sum(Zn)/length(Zn), 1f0; atol=0.15f0)

    # Reproducible through an explicit generator, and the Dims form agrees
    @test rand(MersenneTwister(1), u, 2, 3) == rand(MersenneTwister(1), u, (2, 3))
end

@testset "backward compatibility" begin
    X = randn(Float32, 1, 1, 4, nb)
    flow = free_flow(); flow(X)

    # `base=nothing` (the default) is exactly `StandardNormal(μ, σ)`
    @test log_likelihood(X) == log_likelihood(X; base=StandardNormal())
    @test log_likelihood(X; σ=2f0) == log_likelihood(X; base=StandardNormal(0f0, 2f0))
    @test log_likelihood(X; μ=1f0, normalized=true) ==
          log_likelihood(X; base=StandardNormal(1f0, 1f0), normalized=true)
    @test log_likelihood_per_sample(X; σ=2f0) ==
          log_likelihood_per_sample(X; base=StandardNormal(0f0, 2f0))
    @test log_likelihood(X, flow) == log_likelihood(X, flow; base=StandardNormal())
    @test log_likelihood_per_sample(X, flow) ==
          log_likelihood_per_sample(X, flow; base=StandardNormal())
    @test ∇log_likelihood(X; σ=2f0) == ∇log_likelihood(X; base=StandardNormal(0f0, 2f0))

    Z = randn(Float32, 1, 1, 4, nb)
    @test inverse_and_log_likelihood(Z, flow)[2] ==
          inverse_and_log_likelihood(Z, flow; base=StandardNormal())[2]

    # `base` and `μ`/`σ` together is a contradiction, not a precedence question
    @test_throws ArgumentError log_likelihood(X; base=BoxUniform(B), σ=2f0)
    @test_throws ArgumentError log_likelihood(X, flow; base=StandardNormal(), μ=1f0)
    @test_throws ArgumentError log_likelihood(X; base=:gaussian)

    # The closed forms stay Gaussian-only, and say so
    @test_throws ArgumentError ∇log_likelihood(X; base=BoxUniform(B))
    @test_throws ArgumentError Hlog_likelihood(X; base=BoxUniform(B))
end

@testset "preserves_box trait" begin
    @test preserves_box(SplineLayer(4; bound=B), B)
    @test preserves_box(SplineLayer(4; bound=B), B + 1f0)     # a wider box still works
    @test !preserves_box(SplineLayer(4; bound=B), B - 1f0)    # a narrower one does not

    # Per instance, not per type: `mix=true` carries a Conv1x1
    @test preserves_box(CouplingLayerSpline(4, 16; bound=B, mix=false), B)
    @test !preserves_box(CouplingLayerSpline(4, 16; bound=B, mix=true), B)

    # A circular spline is a bijection of the torus [-B, B), so the bounds must match exactly
    @test preserves_box(SplineLayer(4; spline=:circular, bound=B), B)
    @test !preserves_box(SplineLayer(4; spline=:circular, bound=B), B + 1f0)

    for L in (ActNorm(4), Conv1x1(4), CouplingLayerGlow(4, 16), TanhBijector())
        @test !preserves_box(L, B)
        @test box_violation(L, B) isa String
    end

    @test preserves_box(box_flow(), B)
    @test isnothing(box_violation(box_flow(), B))
    @test !preserves_box(free_flow(), B)
    @test !preserves_box(reverse(box_flow()), B - 1f0)
    @test preserves_box(reverse(box_flow()), B)
    @test !preserves_box(AugmentedFlow(free_flow(; k=8), 4), B)
end

@testset "the guards fire" begin
    X = inbox(4, nb)
    u = BoxUniform(B)

    # A network that leaves the box is a modelling error, reported as one rather than as a
    # quietly wrong number
    bad = free_flow(); bad(X)
    @test_throws ArgumentError log_likelihood(X, bad; base=u)
    @test_throws ArgumentError log_likelihood_per_sample(X, bad; base=u)
    @test_throws ArgumentError inverse_and_log_likelihood(X, bad; base=u)
    @test_throws ArgumentError inverse_and_log_likelihood_per_sample(X, bad; base=u)
    @test_throws ArgumentError check_latent_support(bad, u)

    # ... and the message names the layers responsible
    msg = try; check_latent_support(bad, u); catch e; e.msg; end
    @test occursin("ActNorm", msg)
    @test occursin("Conv1x1", msg)
    @test occursin("CouplingLayerGlow", msg)

    # Bounds have to nest the right way: B_base >= B_spline
    good = box_flow()
    @test isnothing(check_latent_support(good, u))
    @test isnothing(check_latent_support(good, BoxUniform(B + 1f0)))
    @test_throws ArgumentError check_latent_support(good, BoxUniform(B - 1f0))
    @test occursin("bound", try; check_latent_support(good, BoxUniform(B - 1f0)); catch e; e.msg; end)

    # An unbounded base imposes nothing, so every network stays usable
    @test isnothing(check_latent_support(bad, StandardNormal()))
    @test log_likelihood(X, bad; base=StandardNormal()) isa Float32

    # An AugmentedFlow appends Gaussian noise, so a bounded base has no meaning on it
    A = AugmentedFlow(free_flow(; k=8), 4)
    @test_throws ArgumentError log_likelihood_importance(X, A; base=u)
end

@testset "identity flow is exactly uniform" begin
    k = 4
    flow = box_flow(; k=k)                     # identity_init=true by default
    X = inbox(k, nb)

    Z, logdet = flow.forward(X; logdet=:sample)
    @test Z ≈ X
    @test all(l -> isapprox(l, 0f0; atol=1f-5), logdet)

    u = BoxUniform(B)
    f = log_likelihood_per_sample(X, flow; base=u, normalized=true)
    @test all(l -> isapprox(l, Float32(-k*log(2B)); atol=1f-5), f)
    @test isapprox(log_likelihood(X, flow; base=u, normalized=true),
                   Float32(-k*log(2B)); atol=1f-5)

    # Without the constant the score is the log-determinant alone, which is the workaround
    # this feature replaces
    @test log_likelihood_per_sample(X, flow; base=u) ≈ forward_per_sample(X, flow)[2]
end

@testset "per-sample and aggregate agree" begin
    k = 4
    X = inbox(k, nb)
    for (net, base) in ((box_flow(; k=k), BoxUniform(B)),
                        (box_flow(; k=k), StandardNormal()),
                        (free_flow(; k=k), StandardNormal(0f0, 2f0)))
        perturb!(net, 0.2f0)
        net(X)
        for normalized in (false, true)
            f = log_likelihood_per_sample(X, net; base=base, normalized=normalized)
            @test length(f) == nb
            @test sum(f)/nb ≈ log_likelihood(X, net; base=base, normalized=normalized) rtol=1f-4
        end
    end

    # Same on the generative path
    net = box_flow(); perturb!(net, 0.2f0)
    Z = inbox(4, nb)
    _, agg = inverse_and_log_likelihood(Z, net; base=BoxUniform(B), normalized=true)
    Xs, per = inverse_and_log_likelihood_per_sample(Z, net; base=BoxUniform(B), normalized=true)
    @test sum(per)/nb ≈ agg rtol=1f-4
    @test all(abs.(Xs) .<= B + 1f-5)
end

@testset "the domain invariant holds at random parameters" begin
    # Identity init trivially stays inside the box, so this only means something once the
    # parameters have moved.
    Random.seed!(23)
    k = 4
    X = inbox(k, 64)

    # Containment holds however far the parameters are pushed: it is a property of the
    # spline's linear tails, not of the parameters being small.
    for scale in (0.3f0, 0.8f0)
        flow = perturb!(box_flow(; k=k), scale)
        Z = flow.forward(X)[1]
        @test all(abs.(Z) .<= B + 1f-5)
        @test all(abs.(flow.inverse(Z)) .<= B + 1f-5)
        @test all(abs.(flow.inverse(rand(BoxUniform(B), 1, 1, k, 64))) .<= B + 1f-5)
    end

    # It is still a bijection of the box, not merely a map into it. Checked at a moderate
    # perturbation: a strongly perturbed spline is near-singular in places, and inverting it
    # loses Float32 precision to the reciprocal of a very large derivative -- conditioning of
    # the spline, unrelated to the domain.
    flow = perturb!(box_flow(; k=k), 0.3f0)
    @test flow.inverse(flow.forward(X)[1]) ≈ X rtol=1f-3
end

@testset "normalization by quadrature" begin
    # With a box-preserving chain and a matching uniform base the density integrates to 1 at
    # *any* parameter value, not only near the identity -- a stronger claim than the Gaussian
    # case admits, where the flow's tails always sit partly outside any finite grid. What is
    # left is the quadrature error itself, so the parameters are perturbed enough to make the
    # density non-trivial but not so far that the spline develops peaks a midpoint rule
    # cannot resolve.
    Random.seed!(31)
    d, ngrid = 2, 300
    u = BoxUniform(B)

    h = 2*Float64(B)/ngrid
    grid = collect(range(-Float64(B) + h/2, Float64(B) - h/2; length=ngrid))
    # The whole grid as one batch: `log_likelihood_per_sample` scores each point separately.
    pts = reshape(reduce(hcat, [Float32[x1, x2] for x1 in grid for x2 in grid]),
                  1, 1, d, ngrid^2)
    mass(net) = sum(exp.(Float64.(log_likelihood_per_sample(pts, net; base=u,
                                                            normalized=true)))) * h^2

    @test isapprox(mass(perturb!(box_flow(; k=d, nh=8), 0.3f0)), 1.0; atol=1e-3)
    # An elementwise spline, same claim
    @test isapprox(mass(perturb!(InvertibleChain(SplineLayer(d; bound=B, logdet=true)), 0.3f0)),
                   1.0; atol=1e-3)
    # And at the identity initialization it is the flat -d·log(2B), integrating to 1 exactly
    @test isapprox(mass(box_flow(; k=d, nh=8)), 1.0; atol=1e-6)

    # For contrast: a Gaussian base over the same finite window loses mass to the tails, so
    # this identity is genuinely a property of the bounded construction.
    gflow = perturb!(box_flow(; k=d, nh=8), 0.3f0)
    gmass = sum(exp.(Float64.(log_likelihood_per_sample(pts, gflow; base=StandardNormal(),
                                                        normalized=true)))) * h^2
    @test gmass < 0.999
end

@testset "circular splines need no torus base of their own" begin
    # A uniform on the torus [-B, B) is the same measure as `BoxUniform(B)`, and a circular
    # spline is a bijection of that torus -- so the existing base covers the periodic case,
    # provided the bounds match exactly, which `preserves_box` already insists on.
    Random.seed!(53)
    k = 4
    flow = perturb!(InvertibleChain(
        CouplingLayerSpline(k, 16; spline=:circular, bound=B, logdet=true),
        SplineLayer(k; spline=:circular, bound=B, logdet=true)), 0.5f0)
    u = BoxUniform(B)

    @test preserves_box(flow, B)
    X = rand(u, 1, 1, k, 64)
    Z = flow.forward(X)[1]
    @test all(-B .<= Z .< B)
    @test isfinite(log_likelihood(X, flow; base=u, normalized=true))

    # A wider base is not the torus the spline acts on, so it is rejected
    @test_throws ArgumentError check_latent_support(flow, BoxUniform(B + 1f0))
    @test occursin("torus", box_violation(flow, B + 1f0))

    # And the density still integrates to 1 over the period
    d, ngrid = 2, 300
    circ = perturb!(InvertibleChain(SplineLayer(d; spline=:circular, bound=B, logdet=true)),
                    0.5f0)
    h = 2*Float64(B)/ngrid
    grid = collect(range(-Float64(B) + h/2, Float64(B) - h/2; length=ngrid))
    pts = reshape(reduce(hcat, [Float32[x1, x2] for x1 in grid for x2 in grid]),
                  1, 1, d, ngrid^2)
    mass = sum(exp.(Float64.(log_likelihood_per_sample(pts, circ; base=u,
                                                       normalized=true)))) * h^2
    @test isapprox(mass, 1.0; atol=1e-3)
end

@testset "training against a bounded base" begin
    Random.seed!(41)
    k = 4
    flow = box_flow(; k=k)
    X = inbox(k, nb)
    u = BoxUniform(B)

    l, grads = Flux.withgradient(m -> -log_likelihood(X, m; base=u), flow)
    @test l isa Float32 && isfinite(l)
    @test any(!isnothing, Flux.trainables(grads[1]))
    @test all(g -> isnothing(g) || all(isfinite, g), Flux.trainables(grads[1]))

    # The base contributes no gradient of its own -- it is constant on the box -- so this is
    # the log-determinant objective, and `normalized` cannot move the gradient either
    _, grads_ld = Flux.withgradient(m -> -sum(forward_per_sample(X, m)[2])/nb, flow)
    @test all(a ≈ b for (a, b) in zip(Flux.trainables(grads[1]), Flux.trainables(grads_ld[1])))

    _, grads_n = Flux.withgradient(m -> -log_likelihood(X, m; base=u, normalized=true), flow)
    @test all(a == b for (a, b) in zip(Flux.trainables(grads[1]), Flux.trainables(grads_n[1])))

    # Per-example weights still reach the layers with a bounded base
    w = rand(Float32, nb)
    _, wg = Flux.withgradient(m -> -sum(w .* log_likelihood_per_sample(X, m; base=u)), flow)
    @test any(!isnothing, Flux.trainables(wg[1]))

    # And a few optimizer steps run without leaving the box or producing a NaN
    opt_state = Flux.setup(Adam(1f-3), flow)
    for _ = 1:5
        _, g = Flux.withgradient(m -> -log_likelihood(X, m; base=u), flow)
        Flux.update!(opt_state, flow, g[1])
    end
    @test isfinite(log_likelihood(X, flow; base=u, normalized=true))
    @test all(abs.(flow.forward(X)[1]) .<= B + 1f-5)
end
