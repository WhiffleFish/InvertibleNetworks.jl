# Tests for InvertibleChain: Flux.Chain-style composition with automatic logdet
# accumulation and a ChainRules pullback.

using InvertibleNetworks, Flux, LinearAlgebra, Test, Random

Random.seed!(17)

n_in = 4
n_hidden = 16
batchsize = 4
X = randn(Float32, 8, 8, n_in, batchsize)

# A fresh, identically-initialized flow, already past the ActNorm initialization pass.
function make_flow(; logdet=true)
    Random.seed!(3)
    flow = InvertibleChain(ActNorm(n_in; logdet=logdet),
                           CouplingLayerGlow(n_in, n_hidden; logdet=logdet),
                           ActNorm(n_in; logdet=logdet))
    flow.forward(X)
    return flow
end

# Index of each `get_params` entry in `Flux.trainables` order (the two disagree on
# traversal order, so pair them up by array identity).
function trainables_order(flow)
    positions = IdDict{Any,Int}(p.data => i for (i, p) in enumerate(get_params(flow)))
    return [positions[data] for data in Flux.trainables(flow)]
end

###################################################################################################

@testset "Chain interface" begin
    flow = make_flow()
    @test length(flow) == 3
    @test flow[2] isa CouplingLayerGlow
    @test flow[1] isa ActNorm
    @test flow[1:2] isa InvertibleChain
    @test length(flow[1:2]) == 2
    @test length(collect(flow)) == 3
    @test occursin("InvertibleChain(", sprint(show, flow))
    @test occursin("CouplingLayerGlow", sprint(show, flow))
end

@testset "forward / inverse" begin
    flow = make_flow()
    Z, logdet = flow.forward(X)
    @test size(Z) == size(X)
    @test logdet isa Float32

    # logdet must be the sum of the individual layers' contributions
    Y1, l1 = flow[1].forward(X)
    Y2, l2 = flow[2].forward(Y1)
    Y3, l3 = flow[3].forward(Y2)
    @test Z ≈ Y3
    @test logdet ≈ l1 + l2 + l3

    @test isapprox(norm(flow.inverse(Z) - X)/norm(X), 0f0; atol=1f-5)

    # A chain of logdet=false layers returns the array on its own, like the layers do
    plain = make_flow(; logdet=false)
    @test !plain.logdet
    @test plain.forward(X) isa AbstractArray
    @test isapprox(norm(plain.inverse(plain.forward(X)) - X)/norm(X), 0f0; atol=1f-5)
end

@testset "callable is consistent inside and outside AD" begin
    flow = make_flow()
    Z, logdet = flow.forward(X)

    outside = flow(X)
    @test outside isa Tuple
    @test outside[1] == Z && outside[2] == logdet

    # The whole point: the same objective can be evaluated for training and for reporting.
    objective(m) = ((Zm, l) = m(X); -log_likelihood(Zm) - l)
    @test objective(flow) isa Real                       # would previously throw
    inside = Ref{Any}(nothing)
    Flux.gradient(m -> ((Zm, l) = m(X); inside[] = (typeof(Zm), typeof(l)); sum(Zm) - l), flow)
    @test inside[] == (typeof(Z), typeof(logdet))
end

@testset "type stability" begin
    flow = make_flow()
    @test only(Base.return_types(InvertibleNetworks.forward, Tuple{typeof(X),typeof(flow)})) ==
          Tuple{Array{Float32,4},Float32}
    plain = make_flow(; logdet=false)
    @test only(Base.return_types(InvertibleNetworks.forward, Tuple{typeof(X),typeof(plain)})) ==
          Array{Float32,4}
end

@testset "gradient matches the hand-written backward" begin
    W = randn(Float32, size(X))

    # Reference: the layers' own backward, which bakes in a -1 logdet weight.
    reference = make_flow()
    Z, _ = reference.forward(X)
    clear_grad!(reference)
    reference.backward(copy(W), copy(Z))
    expected = [p.grad for p in get_params(reference)]

    flow = make_flow()
    _, grads = Flux.withgradient(m -> ((Zm, l) = m(X); sum(Zm .* W) - l), flow)
    got = Flux.trainables(grads[1])
    order = trainables_order(flow)
    @test length(got) == length(expected)
    @test all(got[k] == expected[i] for (k, i) in enumerate(order))
end

@testset "Conv1x1 composes inside a chain" begin
    # `InvertibleChain.backward` calls `backward(ΔY, Y, layer)` on every layer, so a layer
    # that only implemented the tuple form was unusable in a chain the moment anything asked
    # for a gradient -- and `Conv1x1` is the standard channel mixer of a Glow-style stack.
    Random.seed!(9)
    flow = InvertibleChain(ActNorm(n_in; logdet=true),
                           Conv1x1(n_in),
                           CouplingLayerGlow(n_in, n_hidden; logdet=true),
                           Conv1x1(n_in),
                           ActNorm(n_in; logdet=true))
    flow.forward(X)

    Z, lgdet = flow.forward(X)
    @test lgdet isa Float32
    @test isapprox(flow.inverse(Z), X; rtol=1f-5)

    clear_grad!(flow)
    ΔX, X̂ = flow.backward(randn(Float32, size(Z)...), copy(Z))
    @test size(ΔX) == size(X)
    @test isapprox(X̂, X; rtol=1f-4)
    @test all(!isnothing(p.grad) for p in get_params(flow))

    loss(m) = -log_likelihood(X, m)
    l, grads = Flux.withgradient(loss, flow)
    @test isfinite(l)
    g = Flux.trainables(grads[1])
    @test length(g) == length(Flux.trainables(flow))
    @test all(gi -> !isnothing(gi) && all(isfinite, gi), g)

    # Directional derivative against a central finite difference, over all parameters at
    # once: this checks the Householder gradients themselves, not just that a method exists.
    base = [copy(p) for p in Flux.trainables(flow)]
    dir = [randn(Float32, size(p)...) for p in base]
    dderiv = sum(dot(gi, di) for (gi, di) in zip(g, dir))

    δ = 1f-3
    for (p, b, u) in zip(Flux.trainables(flow), base, dir); p .= b .+ δ.*u; end
    lp = loss(flow)
    for (p, b, u) in zip(Flux.trainables(flow), base, dir); p .= b .- δ.*u; end
    lm = loss(flow)
    for (p, b) in zip(Flux.trainables(flow), base); p .= b; end
    @test isapprox((lp - lm)/(2δ), dderiv; rtol=5f-2, atol=1f-2)

    # Only the Householder vectors, so a wrong `Conv1x1` gradient cannot hide behind the
    # coupling layers' much larger contribution.
    conv_params = vcat([[C.v1.data, C.v2.data, C.v3.data] for C in (flow[2], flow[4])]...)
    positions = IdDict{Any,Int}(p => i for (i, p) in enumerate(Flux.trainables(flow)))
    idx = [positions[p] for p in conv_params]
    vdir = [randn(Float32, size(p)...) for p in conv_params]
    vderiv = sum(dot(g[i], d) for (i, d) in zip(idx, vdir))

    for (p, b, u) in zip(conv_params, base[idx], vdir); p .= b .+ δ.*u; end
    lp = loss(flow)
    for (p, b, u) in zip(conv_params, base[idx], vdir); p .= b .- δ.*u; end
    lm = loss(flow)
    for (p, b) in zip(conv_params, base[idx]); p .= b; end
    @test isapprox((lp - lm)/(2δ), vderiv; rtol=5f-2, atol=1f-3)
    @test abs(vderiv) > 1f-3        # the direction actually moves the loss

    # And per-sample weights reach the chain the same way
    w = rand(Float32, batchsize)
    _, wgrads = Flux.withgradient(m -> -sum(w .* log_likelihood_per_sample(X, m)), flow)
    @test all(gi -> !isnothing(gi) && all(isfinite, gi), Flux.trainables(wgrads[1]))
end

@testset "gradient honors any logdet weight" begin
    # `backward(ΔZ) = A(ΔZ) - B` is affine in the cotangent, so recover A and B from two
    # passes and require the AD gradient to equal `A + w*B` for whatever weight the loss
    # gives the log-determinant. The layers only ever compute `w = -1` on their own.
    W = randn(Float32, size(X))
    reference = make_flow()
    Z, _ = reference.forward(X)
    clear_grad!(reference); reference.backward(copy(W), copy(Z))
    A_minus_B = [p.grad for p in get_params(reference)]
    clear_grad!(reference); reference.backward(zero(W), copy(Z))
    minus_B = [p.grad for p in get_params(reference)]
    A = [a .- b for (a, b) in zip(A_minus_B, minus_B)]
    B = [-b for b in minus_B]

    for w in (-1f0, 0f0, -3f0, -0.25f0, 2f0)
        flow = make_flow()
        _, grads = Flux.withgradient(m -> ((Zm, l) = m(X); sum(Zm .* W) + w*l), flow)
        got = Flux.trainables(grads[1])
        order = trainables_order(flow)
        for (k, i) in enumerate(order)
            want = A[i] .+ w .* B[i]
            @test isapprox(got[k], want; rtol=1f-5)
        end
    end
end

@testset "gradients do not leak state" begin
    flow = make_flow()
    objective(m) = ((Zm, l) = m(X); sum(Zm) - l)

    # Repeated calls must agree ...
    _, g1 = Flux.withgradient(objective, flow)
    _, g2 = Flux.withgradient(objective, flow)
    @test all(a == b for (a, b) in zip(Flux.trainables(g1[1]), Flux.trainables(g2[1])))

    # ... and leftover `p.grad` must not contaminate the result. Conv1x1 accumulates into
    # `p.grad` where the other layers overwrite, so this is a real hazard.
    for p in get_params(flow)
        p.grad = fill!(similar(p.data), 1f6)
    end
    _, g3 = Flux.withgradient(objective, flow)
    @test all(a == b for (a, b) in zip(Flux.trainables(g1[1]), Flux.trainables(g3[1])))
end

@testset "Flux optimiser integration" begin
    flow = make_flow()
    objective(m) = ((Zm, l) = m(X); -log_likelihood(Zm) - l)

    opt_state = Flux.setup(Descent(1f-2), flow)
    _, grads = Flux.withgradient(objective, flow)
    params_pre = deepcopy(Flux.trainables(flow))
    Flux.update!(opt_state, flow, grads[1])
    @test all(a != b for (a, b) in zip(params_pre, Flux.trainables(flow)))

    # And a short training loop should actually make progress
    flow2 = make_flow()
    state2 = Flux.setup(Adam(1f-3), flow2)
    loss0 = objective(flow2)
    for _ = 1:25
        _, g = Flux.withgradient(objective, flow2)
        Flux.update!(state2, flow2, g[1])
    end
    @test objective(flow2) < loss0
end

@testset "composition and conversion" begin
    # Chains nest
    inner = InvertibleChain(ActNorm(n_in; logdet=true), CouplingLayerGlow(n_in, n_hidden; logdet=true))
    outer = InvertibleChain(inner, ActNorm(n_in; logdet=true))
    Z, logdet = outer.forward(X)
    @test logdet isa Float32
    @test isapprox(norm(outer.inverse(Z) - X)/norm(X), 0f0; atol=1f-5)
    @test length(get_params(outer)) == length(get_params(inner)) + 2

    # Functors reconstruction (gpu/f32/f64 all go through this path)
    flow = make_flow()
    @test typeof(Flux.f32(flow)) === typeof(flow)
    flow64 = Flux.f64(flow)
    @test length(flow64) == length(flow)
    @test flow64.logdet
    Z64, logdet64 = flow64.forward(Float64.(X))
    @test eltype(Z64) == Float64
    @test logdet64 isa Float64
end

@testset "log_likelihood of the data" begin
    flow = make_flow()
    Z, logdet = flow.forward(X)

    # Change of variables: log p_X(X) = log p_Z(Z) + log|det J|
    @test log_likelihood(X, flow) ≈ log_likelihood(Z) + logdet
    @test -log_likelihood(X, flow) ≈ -log_likelihood(Z) - logdet
    @test log_likelihood(X, flow; σ=2f0) ≈ log_likelihood(Z; σ=2f0) + logdet

    # The gradient of the negative log-likelihood is the layers' own backward, which bakes
    # in exactly the -1 logdet weight this objective implies.
    reference = make_flow()
    Zr, _ = reference.forward(X)
    clear_grad!(reference)
    reference.backward(-∇log_likelihood(Zr), Zr)
    expected = IdDict{Any,Any}(p.data => p.grad for p in get_params(reference))
    want = [expected[data] for data in Flux.trainables(reference)]

    flow2 = make_flow()
    _, grads = Flux.withgradient(m -> -log_likelihood(X, m), flow2)
    @test all(a == b for (a, b) in zip(Flux.trainables(grads[1]), want))

    # Also available for the other invertible networks, not just chains
    Random.seed!(5)
    G = NetworkGlow(n_in, n_hidden, 1, 2; logdet=true)
    G(X)
    ZG, ldG = G.forward(X)
    @test log_likelihood(X, G) ≈ log_likelihood(ZG) + ldG
    _, gradsG = Flux.withgradient(m -> -log_likelihood(X, m), G)
    @test all(!isnothing, Flux.trainables(gradsG[1]))

    # Without a log-determinant there is no density to report
    plain = make_flow(; logdet=false)
    @test_throws ArgumentError log_likelihood(X, plain)
end

@testset "log_likelihood integrates to one" begin
    # d = 2 dimensions per sample, so log p_X(x) = log_likelihood(x, flow) - (d/2)log(2π)
    # and the density must integrate to 1 over R². The coupling layers make this a genuine
    # check of the change-of-variables identity rather than a restatement of it.
    Random.seed!(4)
    d = 2
    flow = InvertibleChain(ActNorm(d; logdet=true),
                           CouplingLayerGlow(d, 8; logdet=true, k1=1, k2=1, p1=0, p2=0),
                           ActNorm(d; logdet=true),
                           CouplingLayerGlow(d, 8; logdet=true, k1=1, k2=1, p1=0, p2=0))
    flow(randn(Float32, 1, 1, d, 64))          # initialize ActNorm from a real batch

    normalizer = -(d/2)*log(2π)
    logpdf(x1, x2) = Float64(log_likelihood(reshape(Float32[x1, x2], 1, 1, d, 1), flow)) + normalizer

    L, ngrid = 10.0, 100
    h = 2L/ngrid
    grid = range(-L + h/2, L - h/2; length=ngrid)
    mass = sum(exp(logpdf(x1, x2)) for x1 in grid for x2 in grid) * h^2
    @test isapprox(mass, 1.0; atol=1e-3)

    # `normalized=true` folds that constant in, so no manual correction is needed
    normalized(x1, x2) = exp(Float64(log_likelihood(reshape(Float32[x1, x2], 1, 1, d, 1),
                                                    flow; normalized=true)))
    @test isapprox(sum(normalized(x1, x2) for x1 in grid for x2 in grid) * h^2, 1.0; atol=1e-3)
end

@testset "optional normalizing constant" begin
    flow = make_flow()

    @test log_likelihood(X, flow; normalized=true) ≈
          log_likelihood(X, flow) + gaussian_lognorm(X, 1f0)
    @test log_likelihood(X; normalized=true) ≈ log_likelihood(X) + gaussian_lognorm(X, 1f0)
    @test log_likelihood(X; σ=2f0, normalized=true) ≈
          log_likelihood(X; σ=2f0) + gaussian_lognorm(X, 2f0)

    # d elements per sample, and the constant keeps the eltype
    d = length(X) ÷ size(X, ndims(X))
    @test gaussian_lognorm(X, 1f0) ≈ Float32(-d/2 * log(2π))
    @test log_likelihood(X, flow; normalized=true) isa Float32

    # It is a constant in the parameters, so gradients must not move
    _, plain_grads = Flux.withgradient(m -> -log_likelihood(X, m), make_flow())
    _, norm_grads = Flux.withgradient(m -> -log_likelihood(X, m; normalized=true), make_flow())
    @test all(a == b for (a, b) in zip(Flux.trainables(plain_grads[1]),
                                       Flux.trainables(norm_grads[1])))

    # Added once per sample rather than averaged, so a batch agrees with per-sample calls
    batched = log_likelihood(X, flow; normalized=true)
    per_sample = [log_likelihood(X[:, :, :, i:i], flow; normalized=true) for i in 1:size(X, 4)]
    @test isapprox(batched, sum(per_sample)/length(per_sample); rtol=1f-5)
end

@testset "per-sample log_likelihood" begin
    flow = make_flow()
    scores = log_likelihood_per_sample(X, flow)

    @test length(scores) == size(X, ndims(X))
    @test eltype(scores) == Float32

    # Unaggregated form of the scalar version
    @test isapprox(sum(scores)/length(scores), log_likelihood(X, flow); rtol=1f-5)

    # Each entry is the scalar call on that one sample. The fused path reduces over the
    # batch last instead of never, so the agreement is up to floating point, not exact.
    @test isapprox(scores, [log_likelihood(X[:, :, :, i:i], flow) for i in eachindex(scores)];
                   rtol=1f-4)

    # The whole point: they differ from one another
    @test length(unique(scores)) == length(scores)

    @test isapprox(log_likelihood_per_sample(X, flow; normalized=true),
                   scores .+ gaussian_lognorm(X[:, :, :, 1:1], 1f0); rtol=1f-5)

    # Available for the other networks too
    Random.seed!(5)
    G = NetworkGlow(n_in, n_hidden, 1, 2; logdet=true)
    G(X)
    @test isapprox(sum(log_likelihood_per_sample(X, G))/size(X, 4), log_likelihood(X, G); rtol=1f-5)

    # A fresh network must be initialized from the whole batch, not from sample 1
    Random.seed!(3)
    fresh = InvertibleChain(ActNorm(n_in; logdet=true),
                            CouplingLayerGlow(n_in, n_hidden; logdet=true),
                            ActNorm(n_in; logdet=true))
    fresh_scores = log_likelihood_per_sample(X, fresh)
    @test fresh[1].s.data ≈ make_flow()[1].s.data
    @test isapprox(fresh_scores, scores; rtol=1f-5)
end

@testset "per-sample log_likelihood is differentiable" begin
    # `sum` of the per-sample values is `batchsize` times the aggregate, so the gradients
    # must agree up to that factor. An absolute tolerance is needed because some entries
    # (the last ActNorm's bias) are zero to within Float32 noise.
    B = size(X, ndims(X))
    _, per_sample_grads = Flux.withgradient(m -> sum(log_likelihood_per_sample(X, m)), make_flow())
    _, scaled_grads = Flux.withgradient(m -> B*log_likelihood(X, m), make_flow())
    @test all(!isnothing, Flux.trainables(per_sample_grads[1]))
    @test all(isapprox(a, b; rtol=1f-3, atol=1f-4)
              for (a, b) in zip(Flux.trainables(per_sample_grads[1]),
                                Flux.trainables(scaled_grads[1])))

    # The reason to want this: per-example weights
    weights = rand(Float32, B)
    _, weighted = Flux.withgradient(m -> -sum(weights .* log_likelihood_per_sample(X, m)), make_flow())
    @test all(!isnothing, Flux.trainables(weighted[1]))
end

@testset "per-sample log_likelihood is a single pass" begin
    flow = make_flow()

    # The per-sample vector is what the scalar log-determinant is the mean of, from the
    # same pass -- not one pass per sample
    Z, lgdet = flow.forward(X)
    Z_ps, lgdet_ps = flow.forward(X; logdet=:sample)
    @test Z_ps ≈ Z
    @test length(lgdet_ps) == batchsize
    @test isapprox(sum(lgdet_ps)/batchsize, lgdet; rtol=1f-5)

    @test supports_per_sample_logdet(flow)
    @test isapprox(log_likelihood_per_sample(X, flow),
                   log_likelihood_per_sample(Z) .+ lgdet_ps; rtol=1f-4)

    # A network whose layers only report a batch average falls back to the loop, which is
    # slower but still correct
    Random.seed!(5)
    G = NetworkGlow(n_in, n_hidden, 1, 2; logdet=true)
    G(X)
    @test !supports_per_sample_logdet(G)
    @test isapprox(sum(log_likelihood_per_sample(X, G))/batchsize, log_likelihood(X, G);
                   rtol=1f-5)

    # A chain that accumulates nothing has no per-sample log-determinant to report
    plain = InvertibleChain(ActNorm(n_in), CouplingLayerGlow(n_in, n_hidden))
    plain(X)
    @test_throws ArgumentError plain.forward(X; logdet=:sample)
    @test_throws ArgumentError flow.forward(X; logdet=:mean)
end

@testset "per-example weights in the loss" begin
    # The batch-averaged pullback rescales one log-determinant gradient for the whole batch,
    # which cannot express per-example weights; check the fused path against the loop.
    weights = rand(Float32, batchsize)
    _, fused = Flux.withgradient(m -> -sum(weights .* log_likelihood_per_sample(X, m)),
                                 make_flow())
    _, loop = Flux.withgradient(make_flow()) do m
        -sum(weights[i]*log_likelihood(X[:, :, :, i:i], m) for i in 1:batchsize)
    end
    @test all(!isnothing, Flux.trainables(fused[1]))
    @test all(isapprox(a, b; rtol=1f-2, atol=1f-4)
              for (a, b) in zip(Flux.trainables(fused[1]), Flux.trainables(loop[1])))

    # A loss that uses only the log-determinant leaves no cotangent on Z at all
    _, logdet_only = Flux.withgradient(m -> sum(weights .* forward_per_sample(X, m)[2]),
                                       make_flow())
    @test all(!isnothing, Flux.trainables(logdet_only[1]))
end

@testset "inverse reports its log-determinant" begin
    flow = make_flow()
    Z, lgdet = flow.forward(X)

    X_, lgdet_inv = flow.inverse(Z; logdet=true)
    @test isapprox(X_, X; rtol=1f-4)
    @test isapprox(lgdet_inv, -lgdet; rtol=1f-5)

    X_ps, lgdet_ps = flow.inverse(Z; logdet=:sample)
    @test isapprox(X_ps, X; rtol=1f-4)
    @test isapprox(sum(lgdet_ps)/batchsize, lgdet_inv; rtol=1f-5)

    # Generating a sample and scoring it is one pass, not two
    Zr = randn(Float32, size(X)...)
    Xg, logp = inverse_and_log_likelihood(Zr, flow)
    @test isapprox(logp, log_likelihood(Xg, flow); rtol=1f-4)

    Xg_ps, logp_ps = inverse_and_log_likelihood_per_sample(Zr, flow)
    @test Xg_ps ≈ Xg
    @test isapprox(logp_ps, log_likelihood_per_sample(Xg_ps, flow); rtol=1f-4)
    @test isapprox(sum(logp_ps)/batchsize, logp; rtol=1f-5)

    Xg_n, logp_n = inverse_and_log_likelihood(Zr, flow; normalized=true)
    @test isapprox(logp_n, log_likelihood(Xg_n, flow; normalized=true); rtol=1f-4)

    plain = InvertibleChain(ActNorm(n_in), CouplingLayerGlow(n_in, n_hidden))
    plain(X)
    @test_throws ArgumentError plain.inverse(X; logdet=true)
end

@testset "lazily-initialized layers" begin
    # A flow used generatively before it has ever seen data has no map yet: the ActNorm
    # parameters are defined by the statistics of the first forward batch, so an `inverse`
    # that silently used s=1, b=0 would change the map on the first loss evaluation.
    Random.seed!(3)
    fresh = InvertibleChain(ActNorm(n_in; logdet=true),
                            CouplingLayerGlow(n_in, n_hidden; logdet=true),
                            ActNorm(n_in; logdet=true))
    @test needs_init(fresh)
    @test_throws ArgumentError fresh.inverse(X)

    init!(fresh, X)
    @test !needs_init(fresh)
    Z, _ = fresh.forward(X)
    @test isapprox(fresh.inverse(Z), X; rtol=1f-5)

    # Same statistics as a flow initialized by a forward pass, and idempotent
    s = copy(fresh[1].s.data)
    init!(fresh, 100f0*X .+ 5f0)
    @test fresh[1].s.data == s
    @test fresh[1].s.data ≈ make_flow()[1].s.data

    # A reversed layer normalizes on the other pass, so the same rule applies mirrored:
    # `rev.forward` is the normalizing direction, `rev.inverse` is undefined before it
    rev = reverse(ActNorm(n_in; logdet=true))
    @test needs_init(rev)
    @test_throws ArgumentError rev.inverse(X)
    rev.forward(X)
    @test !needs_init(rev)
end

@testset "coupling layers need a channel split" begin
    @test_throws ArgumentError CouplingLayerGlow(1, n_hidden)
    @test_throws ArgumentError ConditionalLayerGlow(1, 2, n_hidden)
    @test CouplingLayerGlow(2, n_hidden) isa CouplingLayerGlow
end

@testset "unsupported interfaces error clearly" begin
    flow = make_flow()
    Z, _ = flow.forward(X)
    @test_throws ArgumentError flow.backward(copy(Z), copy(Z); set_grad=false)
    @test_throws ArgumentError flow.jacobian(X, get_params(flow), X)
end
