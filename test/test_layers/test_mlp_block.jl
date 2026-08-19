# Test the dense conditioner block, and the vector-shaped coupling layer it enables
using LinearAlgebra, InvertibleNetworks, Test, Random, Flux
using InvertibleNetworks: block_forward, block_forward_save, block_backward, channel_halves

Random.seed!(7)

###################################################################################################
# Gradient of the block itself, against a finite difference of its own forward pass

@testset "MLPBlock gradients" begin
    d_in, n_hidden, batchsize = 6, 12, 4

    for fan in [true, false]
        B = MLPBlock(d_in, n_hidden; fan=fan)
        X = randn(Float32, d_in, batchsize)
        ΔY = randn(Float32, size(B.forward(X))...)

        # Adjoint test: <ΔY, J dX> == <J' ΔY, dX>
        dX = randn(Float32, d_in, batchsize)
        ΔX = B.backward(ΔY, X)

        h = 1f-3
        Yp = B.forward(X .+ (h/2) .* dX)
        Ym = B.forward(X .- (h/2) .* dX)
        JdX = (Yp .- Ym) ./ h
        @test isapprox(dot(ΔY, JdX), dot(ΔX, dX); rtol=1f-2)
    end
end

###################################################################################################
# Parameter gradients by finite difference of a scalar objective

@testset "MLPBlock parameter gradients" begin
    d_in, n_hidden, batchsize = 4, 8, 3
    B = MLPBlock(d_in, n_hidden; fan=true)
    X = randn(Float32, d_in, batchsize)
    ΔY = randn(Float32, 2*d_in, batchsize)

    f(B) = dot(ΔY, block_forward(X, B))
    B.backward(ΔY, X)

    h = 1f-3
    for (p, name) in [(B.W1, :W1), (B.W2, :W2), (B.W3, :W3), (B.b1, :b1), (B.b2, :b2)]
        dp = randn(Float32, size(p.data)...)
        p0 = copy(p.data)
        p.data .= p0 .+ (h/2) .* dp; fp = f(B)
        p.data .= p0 .- (h/2) .* dp; fm = f(B)
        p.data .= p0
        @test isapprox((fp - fm)/h, dot(p.grad, dp); rtol=1f-2)
    end
end

###################################################################################################
# The saved state has to agree with the plain forward pass

@testset "MLPBlock saved state" begin
    B = MLPBlock(6, 10; fan=true)
    X = randn(Float32, 6, 5)
    @test isapprox(block_forward(X, B), InvertibleNetworks.block_output(block_forward_save(X, B)))
end

###################################################################################################
# Ranks above 2: the block flattens the leading dimensions

@testset "MLPBlock flattens leading dimensions" begin
    nx, n_in, batchsize = 8, 3, 4
    B = MLPBlock(nx*n_in, 16; d_out=nx*2*n_in, fan=true)
    X = randn(Float32, nx, n_in, batchsize)
    Y = B.forward(X)
    @test size(Y) == (nx, 2*n_in, batchsize)

    # A mismatched flattened size is reported as such, not as a BLAS error
    @test_throws DimensionMismatch B.forward(randn(Float32, nx+1, n_in, batchsize))
end

###################################################################################################
# A coupling layer on plain (dim, batch) vectors

@testset "Vector-shaped Glow coupling" begin
    dim, n_hidden, batchsize = 8, 16, 6
    L = CouplingLayerGlow(dim, n_hidden; logdet=true, dense=true)
    X = randn(Float32, dim, batchsize)

    Y, logdet = L.forward(X)
    @test size(Y) == size(X)
    @test isapprox(norm(X - L.inverse(Y))/norm(X), 0f0; atol=1f-5)

    # Per-sample log-determinant averages to the scalar one
    _, v = L.forward(X; logdet=:sample)
    @test length(v) == batchsize
    @test isapprox(sum(v)/batchsize, logdet; rtol=1f-5)
end

@testset "Vector-shaped Glow chain trains" begin
    dim, n_hidden, batchsize = 6, 16, 32
    flow = InvertibleChain(ActNorm(dim; logdet=true),
                           CouplingLayerGlow(dim, n_hidden; logdet=true, dense=true),
                           ActNorm(dim; logdet=true),
                           CouplingLayerGlow(dim, n_hidden; logdet=true, dense=true))
    X = randn(Float32, dim, batchsize)
    flow(X)   # initialize ActNorm

    # The scalar log-determinant is the mean of the per-sample vector
    Z, ld = flow.forward(X)
    _, v = flow.forward(X; logdet=:sample)
    @test isapprox(ld, sum(v)/batchsize; rtol=1f-5)
    @test isapprox(norm(X - flow.inverse(Z))/norm(X), 0f0; atol=1f-5)

    # One optimizer step must decrease the loss
    loss(m) = -log_likelihood(X, m)
    l0 = loss(flow)
    opt_state = Flux.setup(Adam(1f-3), flow)
    for _ = 1:5
        _, grads = Flux.withgradient(loss, flow)
        Flux.update!(opt_state, flow, grads[1])
    end
    @test loss(flow) < l0
end
