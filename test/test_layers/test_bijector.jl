# Tests for the bounded-support elementwise bijectors.

using InvertibleNetworks, LinearAlgebra, Test, Random

Random.seed!(23)

nx = 1
n_in = 4
batchsize = 8
h = 1f-3

# Both flavors, with bounds that are not the defaults, so a dropped span shows up.
cases = (("SigmoidBijector", (; kw...) -> SigmoidBijector(; low=-2f0, high=3f0, kw...), -2f0, 3f0),
         ("TanhBijector", (; kw...) -> TanhBijector(; low=-1f0, high=2f0, kw...), -1f0, 2f0))

for (name, make, low, high) in cases
    @testset "$name" begin
        B = make(; logdet=true)
        X = low .+ (high - low)*(0.1f0 .+ 0.8f0*rand(Float32, nx, n_in, batchsize))

        # Invertibility, and the inverse log-determinant is the negative of the forward one
        Z, lgdet = B.forward(X)
        @test isapprox(B.inverse(Z), X; rtol=1f-4)
        X_, lgdet_inv = B.inverse(Z; logdet=true)
        @test isapprox(X_, X; rtol=1f-4)
        @test isapprox(lgdet_inv, -lgdet; rtol=1f-4)

        # The point of the layer: the inverse lands inside the interval whatever it is given
        @test all(low .<= B.inverse(20f0*randn(Float32, size(X)...)) .<= high)

        # The map is elementwise, so the Jacobian is diagonal and a finite difference of the
        # forward map gives the log-determinant directly
        num = sum(log.(abs.((B.forward(X .+ h)[1] .- B.forward(X .- h)[1])/(2h))))/batchsize
        @test isapprox(lgdet, num; rtol=1f-2)

        # Per-sample log-determinant averages to the scalar one
        Zs, per_sample = B.forward(X; logdet=:sample)
        @test Zs ≈ Z
        @test length(per_sample) == batchsize
        @test isapprox(sum(per_sample)/batchsize, lgdet; rtol=1f-4)

        # `backward` differentiates `<ΔZ, Z> - logdet`, which has no parameters here but does
        # depend on where the sample sits
        ΔZ = randn(Float32, size(Z)...)
        ΔX, X_rec = B.backward(copy(ΔZ), copy(Z))
        @test isapprox(X_rec, X; rtol=1f-4)

        function objective(x)
            z, l = B.forward(x)
            return sum(ΔZ .* z) - l
        end
        grad = similar(X)
        for i in eachindex(X)
            Xp = copy(X); Xp[i] += h
            Xm = copy(X); Xm[i] -= h
            grad[i] = (objective(Xp) - objective(Xm))/(2h)
        end
        @test isapprox(ΔX, grad; rtol=1f-2, atol=1f-3)

        # No log-determinant unless asked for
        plain = make()
        @test plain.forward(X) isa AbstractArray
        @test isapprox(plain.inverse(plain.forward(X)), X; rtol=1f-4)

        @test isempty(get_params(B))
        @test supports_per_sample_logdet(B)
    end
end

@testset "bijector arguments" begin
    @test_throws ArgumentError SigmoidBijector(; low=1f0, high=0f0)
    @test_throws ArgumentError TanhBijector(; low=0f0, high=0f0)
end

@testset "bounded flow" begin
    X = -0.9f0 .+ 1.8f0*rand(Float32, nx, n_in, batchsize)

    function make_flow()
        Random.seed!(4)
        InvertibleChain(TanhBijector(; logdet=true),
                        ActNorm(n_in; logdet=true),
                        CouplingLayerGlow(n_in, 16; logdet=true, ndims=1))
    end

    flow = make_flow()
    init!(flow, X)
    @test supports_per_sample_logdet(flow)

    # Squashing inside the flow means generated samples are in range by construction, and
    # their density accounts for the squash without any bookkeeping by the caller
    Z = randn(Float32, size(X)...)
    Xg, logp = inverse_and_log_likelihood_per_sample(Z, flow)
    @test all(-1f0 .<= Xg .<= 1f0)
    @test isapprox(logp, log_likelihood_per_sample(Xg, flow); rtol=1f-3)

    scores = log_likelihood_per_sample(X, flow)
    @test isapprox(scores, [log_likelihood(X[:, :, i:i], flow) for i in 1:batchsize];
                   rtol=1f-3)
end
