# State-conditioned spline flows: `CouplingLayerSpline(n_in, n_hidden; n_ctx)` and the
# `InvertibleChain` entry points that thread a context through it.
#
# The context is concatenated onto the channels the conditioner reads and never touches the
# channels the spline transforms, so the layer stays a bijection in `X` at every fixed context.
# What that buys, and what these tests are checking, is that the box-native construction --
# spline couplings with `mix=false` against a `BoxUniform` base -- keeps all three of its
# properties per context: exact normalization, an identity initialization that is exactly the
# uniform density, and no unbounded latent to saturate.

using InvertibleNetworks, Flux, Random, Statistics, Test

const IN = InvertibleNetworks

Random.seed!(0)

# Every test file is included into the same `Main`, so these names have to coexist with the
# rest of the suite: `d`, `nc` and `N` are plain globals elsewhere and so cannot be `const`
# here, and `B` is already a `const` in `test_latent_distributions.jl`, so the box half-width
# is spelled `Bbox` instead. None of this matters when the file is run on its own, which is
# why it is easy to miss.
d, nc, Bbox, N = 2, 5, 3.0f0, 16

boxchain(; n_ctx = nc, depth = 4) = InvertibleChain(
    ntuple(i -> CouplingLayerSpline(d, 8; n_ctx, mix = false, swap = isodd(i),
                                    bound = Bbox, logdet = true), depth))

# Identity init satisfies every invariant trivially, so anything about the *map* has to be
# checked away from it.
perturb!(net, s = 0.3f0) =
    (for p in get_params(net); p.data .+= s .* randn(Float32, size(p.data)); end; net)

base = BoxUniform(Bbox)
X = rand(base, 1, 1, d, N)
Ctx = randn(Float32, 1, 1, nc, N)

@testset "conditional spline flow" begin

@testset "structural" begin
    net = boxchain()
    # The whole point: conditioning does not cost the box invariant, because `preserves_box`
    # answers a question about the spline parameterization and not about where the knots came
    # from. If this ever needs a new method, the layer has stopped being a bijection of the box.
    @test preserves_box(net, Bbox)
    @test check_latent_support(net, base) === nothing
    @test supports_per_sample_logdet(net)
end

@testset "identity init is the magnet, for every context" begin
    net = boxchain()
    lp = log_likelihood_per_sample(X, Ctx, net; base, normalized = true)
    @test all(≈(-d * log(2B); atol = 1e-5), lp)          # exactly uniform on the box
    # ... and it stays uniform when the context changes. A conditioner whose zeroed output layer
    # left any live path from the context would fail here and nowhere else.
    @test lp ≈ log_likelihood_per_sample(X, 5 .* Ctx, net; base, normalized = true)
end

@testset "round trip and per-sample log-determinant" begin
    net = perturb!(boxchain())
    Z, ld = IN.forward(X, Ctx, net; logdet = :sample)
    @test size(Z) == size(X)
    @test all(-Bbox .<= Z .<= Bbox)                            # still a bijection of the box
    @test IN.inverse(Z, Ctx, net) ≈ X atol=1e-4
    _, ld_mean = IN.forward(X, Ctx, net; logdet = true)
    @test mean(ld) ≈ ld_mean atol=1e-5
    _, ld_inv = IN.inverse(Z, Ctx, net; logdet = :sample)
    @test ld_inv ≈ -ld atol=1e-4
end

@testset "the density really depends on the context" begin
    net = perturb!(boxchain())
    lp1 = log_likelihood_per_sample(X, Ctx, net; base, normalized = true)
    lp2 = log_likelihood_per_sample(X, randn(Float32, 1, 1, nc, N), net; base, normalized = true)
    @test !isapprox(lp1, lp2)
end

@testset "normalization is exact at every context" begin
    # Grid quadrature over the box at one fixed context, on a non-identity flow. `∫μ(·|c) = 1`
    # holds by construction for any parameter value, so this is a check on the implementation
    # rather than on training.
    net = perturb!(boxchain(), 0.2f0)
    G = 160
    g = range(-Bbox, Bbox; length = G)
    Xg = reshape(Float32[c == 1 ? x : y for c in 1:2, x in g, y in g], 1, 1, d, G * G)
    c1 = repeat(reshape(Ctx[1, 1, :, 1], 1, 1, nc, 1), 1, 1, 1, G * G)
    lp = log_likelihood_per_sample(Xg, c1, net; base, normalized = true)
    @test sum(exp.(lp)) * (2B / (G - 1))^2 ≈ 1 atol=2e-2
end

@testset "gradient w.r.t. the context, against finite differences" begin
    net = perturb!(boxchain())
    w = randn(Float32, N)
    loss(C) = sum(w .* log_likelihood_per_sample(X, C, net; base, normalized = true))

    g = Flux.gradient(loss, Ctx)[1]
    @test size(g) == size(Ctx)
    @test all(isfinite, g)

    # This is the test that matters. One context feeds all four layers, so its cotangents ADD;
    # an implementation that overwrote instead of accumulating would pass every other test in
    # this file and fail only here, and only at depth > 1.
    h = 1.0f-2
    err = 0.0
    for idx in [(1, 1, 1, 1), (1, 1, 3, 2), (1, 1, nc, N), (1, 1, 2, 7)]
        Cp = copy(Ctx); Cp[idx...] += h
        Cm = copy(Ctx); Cm[idx...] -= h
        fd = (loss(Cp) - loss(Cm)) / (2h)
        err = max(err, abs(fd - g[idx...]) / max(1, abs(fd)))
    end
    @test err < 2e-2
end

@testset "the unconditional path is unchanged" begin
    net = perturb!(InvertibleChain(ntuple(i ->
        CouplingLayerSpline(d, 8; mix = false, swap = isodd(i), bound = Bbox, logdet = true), 4)))
    w = randn(Float32, N)
    l, gs = Flux.withgradient(net) do m
        sum(w .* log_likelihood_per_sample(X, m; base, normalized = true))
    end
    @test isfinite(l)
    @test any(any(!iszero, p) for p in Flux.trainables(gs[1]))
end

@testset "the zero-init dead spot is one optimizer step, not a stall" begin
    net = boxchain()
    loss(m, C) = sum(log_likelihood_per_sample(X, C, m; base, normalized = true))

    # `identity_init` zeroes the conditioner's output convolution, and the same zero that makes
    # the map the identity also blocks the gradient flowing back through it -- so at step 0 the
    # context reports no gradient at all. The conditioner's OWN gradient is nonzero, which is
    # what moves that convolution off zero and unsticks the context pathway on the next step.
    # Documented as a test so it is not mistaken for a bug, or "fixed".
    @test all(iszero, Flux.gradient(C -> loss(net, C), Ctx)[1])
    gθ = Flux.gradient(m -> loss(m, Ctx), net)[1]
    @test any(any(!iszero, p) for p in Flux.trainables(gθ))

    Flux.update!(Flux.setup(Flux.Adam(1.0f-2), net), net, gθ)
    @test any(!iszero, Flux.gradient(C -> loss(net, C), Ctx)[1])
end

@testset "mixed chain: layers with no conditioner ignore the context" begin
    net = InvertibleChain(
        CouplingLayerSpline(d, 8; n_ctx = nc, mix = false, bound = Bbox, logdet = true),
        SplineLayer(d; bound = Bbox, logdet = true),
        CouplingLayerSpline(d, 8; n_ctx = nc, mix = false, swap = true, bound = Bbox, logdet = true))
    @test preserves_box(net, Bbox)
    perturb!(net)
    Z, _ = IN.forward(X, Ctx, net; logdet = :sample)
    @test IN.inverse(Z, Ctx, net) ≈ X atol=1e-4
    @test all(isfinite, Flux.gradient(Ctx) do C
        sum(log_likelihood_per_sample(X, C, net; base, normalized = true))
    end[1])
end

@testset "misuse is rejected loudly" begin
    @test_throws DimensionMismatch IN.forward(X, randn(Float32, 1, 1, nc + 1, N), boxchain())
    @test_throws DimensionMismatch IN.forward(X, Ctx, boxchain(; n_ctx = 0))
    # A chain that reads no context would return a perfectly ordinary unconditional density,
    # and the only symptom would be a context pathway that never learns.
    plain = InvertibleChain(SplineLayer(d; bound = Bbox, logdet = true))
    @test_throws ArgumentError log_likelihood_per_sample(X, Ctx, plain; base, normalized = true)
end

end
