# Tests for AugmentedFlow: a flow run on the data padded with auxiliary latent variables.

using InvertibleNetworks, Flux, LinearAlgebra, Test, Random, Statistics

Random.seed!(23)

n_in = 4
naug = 2
n_tot = n_in + naug
n_hidden = 16
batchsize = 4
X = randn(Float32, 8, 8, n_in, batchsize)

function make_augmented(; naug=naug, σ=1f0)
    Random.seed!(5)
    inner = InvertibleChain(ActNorm(n_in + naug; logdet=true),
                            Conv1x1(n_in + naug),
                            CouplingLayerGlow(n_in + naug, n_hidden; logdet=true),
                            ActNorm(n_in + naug; logdet=true),
                            CouplingLayerGlow(n_in + naug, n_hidden; logdet=true))
    A = AugmentedFlow(inner, naug; σ=σ)
    A.forward(X)      # initialize ActNorm
    return A
end

###################################################################################################

@testset "Construction" begin
    A = make_augmented()
    @test A isa InvertibleNetwork
    @test augmentation_size(A) == naug
    @test occursin("AugmentedFlow", sprint(show, A))
    @test supports_per_sample_logdet(A)

    inner = InvertibleChain(ActNorm(n_tot; logdet=true))
    @test_throws ArgumentError AugmentedFlow(inner, 0)
    @test_throws ArgumentError AugmentedFlow(inner, naug; σ=0f0)

    # `logdet` is a type parameter, so `forward`'s return type is inferable
    inferred = Base.return_types(InvertibleNetworks.forward, (typeof(X), typeof(A)))[1]
    @test inferred == Tuple{Array{Float32,4},Float32}
    plain = AugmentedFlow(A.flow, naug; logdet=false)
    @test Base.return_types(InvertibleNetworks.forward,
                            (typeof(X), typeof(plain)))[1] == Array{Float32,4}
end

@testset "forward shape and stochasticity" begin
    A = make_augmented()
    Z, logdet = A.forward(X)
    @test size(Z) == (8, 8, n_tot, batchsize)
    @test logdet isa Float32

    # Two calls draw different auxiliary noise, so the latent differs
    Z2, _ = A.forward(X)
    @test !isapprox(Z, Z2)

    # ...unless the draw is pinned down
    E = randn(Float32, 8, 8, naug, batchsize)
    Za, la = A.forward(X; ε=E)
    Zb, lb = A.forward(X; ε=E)
    @test Za ≈ Zb
    @test la ≈ lb

    # An explicit rng makes the draw reproducible too
    Zr1, _ = A.forward(X; rng=MersenneTwister(11))
    Zr2, _ = A.forward(X; rng=MersenneTwister(11))
    @test Zr1 ≈ Zr2
end

@testset "inverse recovers the data part" begin
    A = make_augmented()
    E = randn(Float32, 8, 8, naug, batchsize)
    Z, _ = A.forward(X; ε=E)

    X̂ = A.inverse(Z)
    @test size(X̂) == size(X)
    @test isapprox(norm(X̂ - X)/norm(X), 0f0; atol=1f-5)

    # ...and `augmented_inverse` recovers the auxiliary part as well
    X̂, Ê = augmented_inverse(Z, A)
    @test isapprox(norm(X̂ - X)/norm(X), 0f0; atol=1f-5)
    @test isapprox(norm(Ê - E)/norm(E), 0f0; atol=1f-5)

    # Round trip through the pinned forward closes exactly
    Ẑ, _ = A.forward(X̂; ε=Ê)
    @test isapprox(norm(Ẑ - Z)/norm(Z), 0f0; atol=1f-5)
end

@testset "logdet folds in -log q(ε)" begin
    A = make_augmented()
    E = randn(Float32, 8, 8, naug, batchsize)
    Z, logdet = A.forward(X; ε=E)

    # The reported term is the inner flow's log-determinant less log q(ε)
    _, ld_inner = A.flow.forward(cat(X, E; dims=3))
    d = length(E) ÷ batchsize
    log_q = sum(-0.5f0*E.^2)/batchsize - Float32(d/2*log(2π))
    @test logdet ≈ ld_inner - log_q

    # so that log_likelihood(X, A) is the ELBO
    Random.seed!(101); elbo = log_likelihood(X, A; normalized=true)
    Random.seed!(101); Zs, lds = A.forward(X)
    @test elbo ≈ log_likelihood(Zs; normalized=true) + lds
end

@testset "per-sample bound" begin
    A = make_augmented()
    E = randn(Float32, 8, 8, naug, batchsize)

    Z, ld_batch = A.forward(X; ε=E)
    _, ld_sample = A.forward(X; ε=E, logdet=:sample)
    @test ld_sample isa Vector{Float32}
    @test length(ld_sample) == batchsize
    @test sum(ld_sample)/batchsize ≈ ld_batch

    Random.seed!(7); f = log_likelihood_per_sample(X, A; normalized=true)
    Random.seed!(7); fb = log_likelihood(X, A; normalized=true)
    @test length(f) == batchsize
    @test sum(f)/batchsize ≈ fb
end

@testset "inverse log-determinant is the negative of the forward one" begin
    A = make_augmented()
    E = randn(Float32, 8, 8, naug, batchsize)
    Z, ld_fwd = A.forward(X; ε=E)

    _, ld_inv = A.inverse(Z; logdet=true)
    @test ld_inv ≈ -ld_fwd

    _, _, ld_inv_s = augmented_inverse(Z, A; logdet=:sample)
    _, ld_fwd_s = A.forward(X; ε=E, logdet=:sample)
    @test ld_inv_s ≈ -ld_fwd_s

    # so the sampling-path score matches the ELBO at that (x, ε)
    X̂, score = inverse_and_log_likelihood(Z, A; normalized=true)
    @test score ≈ log_likelihood(Z; normalized=true) + ld_fwd
end

@testset "importance-weighted bound" begin
    A = make_augmented()

    # One sample is the plain ELBO
    Random.seed!(13); iw1 = log_likelihood_importance(X, A; nsamples=1, normalized=true)
    Random.seed!(13); elbo = log_likelihood(X, A; normalized=true)
    @test iw1 ≈ elbo

    f = log_likelihood_importance_per_sample(X, A; nsamples=4, normalized=true)
    @test length(f) == batchsize
    @test all(isfinite, f)
    @test log_likelihood_importance(X, A; nsamples=4, normalized=true) isa Float32
    @test_throws ArgumentError log_likelihood_importance(X, A; nsamples=0)

    # The multi-sample bound is at least as tight as the average single-sample one, which is
    # what Jensen guarantees; averaged over draws so the comparison is not a coin flip.
    Random.seed!(31)
    elbos = mean(log_likelihood(X, A; normalized=true) for _ = 1:32)
    iw = mean(log_likelihood_importance(X, A; nsamples=8, normalized=true) for _ = 1:8)
    @test iw >= elbos - 1f-3
end

@testset "backward matches the wrapped flow" begin
    A = make_augmented()
    E = randn(Float32, 8, 8, naug, batchsize)
    Z, _ = A.forward(X; ε=E)
    ΔZ = randn(Float32, size(Z)...)

    ΔX, X̂ = A.backward(copy(ΔZ), copy(Z))
    @test size(ΔX) == size(X)
    @test isapprox(norm(X̂ - X)/norm(X), 0f0; atol=1f-5)

    # The auxiliary slice of the flow's input cotangent is the part that gets dropped
    ΔXA, _ = A.flow.backward(copy(ΔZ), copy(Z))
    @test ΔX ≈ ΔXA[:, :, 1:n_in, :]
end

@testset "AD: gradient of the ELBO" begin
    A = make_augmented()

    # `A(X)` returns (Z, logdet) inside a gradient, like InvertibleChain
    out = A(X)
    @test out isa Tuple && length(out) == 2

    l, grads = Flux.withgradient(m -> -log_likelihood(X, m), A)
    @test isfinite(l)
    g = Flux.trainables(grads[1])
    @test length(g) == length(Flux.trainables(A))
    @test all(gi -> !isnothing(gi) && all(isfinite, gi), g)
    @test any(gi -> norm(gi) > 0, g)

    # The hand-written backward is the gradient of this loss. Compare it against a finite
    # difference along a random parameter direction, seeding before every evaluation so all
    # of them see the same auxiliary draw.
    elbo_loss(m) = (Random.seed!(77); -log_likelihood(X, m))

    base = [copy(p) for p in Flux.trainables(A)]
    dir = [randn(Float32, size(p)...) for p in base]

    Random.seed!(77)
    _, gr = Flux.withgradient(m -> -log_likelihood(X, m), A)
    dderiv = sum(dot(gi, di) for (gi, di) in zip(Flux.trainables(gr[1]), dir))

    δ = 1f-3
    for (p, b, u) in zip(Flux.trainables(A), base, dir); p .= b .+ δ.*u; end
    lp = elbo_loss(A)
    for (p, b, u) in zip(Flux.trainables(A), base, dir); p .= b .- δ.*u; end
    lm = elbo_loss(A)
    for (p, b) in zip(Flux.trainables(A), base); p .= b; end

    @test isapprox((lp - lm)/(2δ), dderiv; rtol=5f-2, atol=1f-2)
end

@testset "AD: per-example weights on the bound" begin
    A = make_augmented()
    w = Float32[0.1, 0.7, 0.2, 1.3]

    Random.seed!(41)
    l, grads = Flux.withgradient(A) do m
        Z, logdet = forward_per_sample(X, m)
        -sum(w .* (log_likelihood_per_sample(Z) .+ logdet))
    end
    @test isfinite(l)
    g = Flux.trainables(grads[1])
    @test all(gi -> !isnothing(gi) && all(isfinite, gi), g)
    @test any(gi -> norm(gi) > 0, g)

    # The importance-weighted bound is differentiable too
    l2, grads2 = Flux.withgradient(m -> -log_likelihood_importance(X, m; nsamples=3), A)
    @test isfinite(l2)
    @test all(gi -> !isnothing(gi) && all(isfinite, gi), Flux.trainables(grads2[1]))
end

@testset "errors" begin
    # A wrapped flow with no log-determinant has nothing for the bound to be built from
    plain = InvertibleChain(ActNorm(n_tot; logdet=false))
    @test_throws ArgumentError AugmentedFlow(plain, naug)

    A = make_augmented()
    @test_throws ArgumentError A.backward(randn(Float32, 8, 8, n_tot, batchsize),
                                          randn(Float32, 8, 8, n_tot, batchsize);
                                          set_grad=false)
    @test_throws ArgumentError A.jacobian(X, get_params(A), X)
end
