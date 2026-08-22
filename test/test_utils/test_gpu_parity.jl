# CPU/GPU parity for the vector-shaped flows: a `1 x 1` spatial map with `d` channels, which is
# how a flow over tabular or action-space data reaches this package.
#
# The other GPU-aware test files pick one device and run the same assertions on it. This one
# needs both at once: it moves an already-built network across and checks that the two devices
# agree, which is what makes the fast paths (the point-spatial GEMM specialization in
# `ResidualBlock`, the indexed bin gather in the spline transforms) safe to rely on.
#
# Without a GPU there is nothing to compare, so every testset here is skipped.

using InvertibleNetworks, LinearAlgebra, Test, Flux, Random

const HAS_GPU = InvertibleNetworks.CUDA.functional()
HAS_GPU || @info "test_gpu_parity: no functional CUDA device, skipping GPU parity tests"

# Float32 GEMMs accumulate in a different order on the two devices, so parity is a tolerance
# and not an equality. These are relative.
const RTOL = 2f-4

rel(a, b) = norm(Array(a) - Array(b)) / max(norm(Array(b)), 1f-20)

@testset "GPU parity: ResidualBlock point-spatial GEMM path" begin
    if !HAS_GPU
        @test_skip "no GPU"
    else
        Random.seed!(51)
        for (n_in, n_hidden, n_out, B, fan, k1, p1) in
                ((17, 64, 46, 512, false, 3, 1),      # the conditioner a spline layer builds
                 (17, 64, 46, 512, false, 1, 0),      # ... with no wasted kernel taps
                 (8,  32, 16, 128, true,  3, 1),      # the conditioner a Glow layer builds
                 (3,  16, 12, 7,   false, 3, 1))      # odd shapes, tiny batch
            RB = ResidualBlock(n_in, n_hidden; n_out=n_out, k1=k1, k2=1, p1=p1, p2=0, fan=fan)
            X  = randn(Float32, 1, 1, n_in, B)
            @test InvertibleNetworks.is_gemm_shaped(X, RB)

            RBg, Xg = gpu(RB), gpu(X)
            Y, Yg = RB.forward(X), RBg.forward(Xg)
            @test rel(Yg, Y) < RTOL

            ΔY = randn(Float32, size(Y))
            ΔX  = RB.backward(ΔY, X)
            ΔXg = RBg.backward(gpu(ΔY), Xg)
            @test rel(ΔXg, ΔX) < RTOL
            for (p, pg) in zip(get_params(RB), get_params(RBg))
                @test rel(pg.grad, p.grad) < RTOL
            end
        end
    end
end

@testset "GPU parity: spline transforms" begin
    if !HAS_GPU
        @test_skip "no GPU"
    else
        Random.seed!(52)
        for kind in (:rqs, :circular, :lrs), nbins in (4, 8)
            spec = InvertibleNetworks.SplineSpec(kind; nbins=nbins, bound=3f0)
            P = n_spline_params(spec)
            B, C = 256, 2
            θ = 0.5f0 .* randn(Float32, 1, 1, C, P, B)
            x = _x = 2f0 .* randn(Float32, 1, 1, C, 1, B)

            kn  = InvertibleNetworks.spline_knots(θ, spec, Val(4))
            kng = InvertibleNetworks.spline_knots(gpu(θ), spec, Val(4))

            y,  lg  = InvertibleNetworks.spline_forward(x, kn, spec, Val(4))
            yg, lgg = InvertibleNetworks.spline_forward(gpu(x), kng, spec, Val(4))
            @test rel(yg, y) < RTOL
            @test rel(lgg, lg) < RTOL

            xr,  _ = InvertibleNetworks.spline_inverse(y, kn, spec, Val(4))
            xrg, _ = InvertibleNetworks.spline_inverse(gpu(y), kng, spec, Val(4))
            @test rel(xrg, xr) < RTOL

            Δy = randn(Float32, size(y))
            gx,  gθ  = InvertibleNetworks.spline_vjp(Δy, -1f0/B, x, kn, spec, Val(4))
            gxg, gθg = InvertibleNetworks.spline_vjp(gpu(Δy), -1f0/B, gpu(x), kng, spec, Val(4))
            @test rel(gxg, gx) < RTOL
            @test rel(gθg, gθ) < RTOL
        end
    end
end

@testset "GPU parity: spline coupling flow, forward/inverse/backward" begin
    if !HAS_GPU
        @test_skip "no GPU"
    else
        Random.seed!(53)
        d, n_ctx, n_hidden, B = 4, 6, 32, 128
        for dense in (false, true), kind in (:rqs, :lrs)
            layers = InvertibleNetworks.Invertible[]
            for i in 1:3
                push!(layers, ActNorm(d; logdet=true))
                push!(layers, CouplingLayerSpline(d, n_hidden; spline=kind, n_ctx=n_ctx,
                                                  nbins=6, logdet=true, swap=isodd(i),
                                                  dense=dense))
            end
            G = InvertibleChain(layers...)

            X = 0.7f0 .* randn(Float32, 1, 1, d, B)
            C = 0.7f0 .* randn(Float32, 1, 1, n_ctx, B)
            init!(G, X, C)                     # ActNorm takes its statistics on the host...
            Gg = gpu(G)                        # ... so the copy carries them across

            Z,  ld  = G.forward(X, C)
            Zg, ldg = Gg.forward(gpu(X), gpu(C))
            @test rel(Zg, Z) < RTOL
            @test isapprox(ldg, ld; rtol=RTOL)

            @test rel(Gg.inverse(Zg, gpu(C)), G.inverse(Z, C)) < RTOL

            ΔZ = randn(Float32, size(Z))
            ΔX,  ΔC,  _ = G.backward(ΔZ, Z, C)
            ΔXg, ΔCg, _ = Gg.backward(gpu(ΔZ), Zg, gpu(C))
            @test rel(ΔXg, ΔX) < RTOL
            @test rel(ΔCg, ΔC) < RTOL
            for (p, pg) in zip(get_params(G), get_params(Gg))
                isnothing(p.grad) && continue
                @test rel(pg.grad, p.grad) < RTOL
            end
        end
    end
end

@testset "GPU parity: parameter plumbing stays on-device" begin
    if !HAS_GPU
        @test_skip "no GPU"
    else
        Random.seed!(54)
        d, B = 4, 64
        G = gpu(InvertibleChain(ActNorm(d; logdet=true),
                                CouplingLayerSpline(d, 16; nbins=6, logdet=true)))
        X = gpu(randn(Float32, 1, 1, d, B))

        # `init!` has to run on-device: ActNorm's statistics come off the batch it is given.
        init!(G, X)
        ps = get_params(G)
        @test all(p -> p.data isa InvertibleNetworks.CUDA.CuArray, ps)

        Z, _ = G.forward(X)
        G.backward(gpu(randn(Float32, size(Z))), Z)
        @test all(p -> isnothing(p.grad) || p.grad isa InvertibleNetworks.CUDA.CuArray, ps)
        # Gradients must match their parameter's shape, or an optimizer step would broadcast
        # into the wrong thing rather than fail.
        for p in ps
            isnothing(p.grad) || @test size(p.grad) == size(p.data)
        end

        clear_grad!(G)
        @test all(p -> isnothing(p.grad), get_params(G))

        # A whole training loop's worth of steps with no host round trip.
        Gt = gpu(InvertibleChain(ActNorm(d; logdet=true),
                                 CouplingLayerSpline(d, 16; nbins=6, logdet=true)))
        init!(Gt, X)
        for _ in 1:3
            Zt, _ = Gt.forward(X)
            Gt.backward(Zt ./ size(Zt, 4), Zt)
            for p in get_params(Gt)
                isnothing(p.grad) && continue
                p.data .-= 1f-3 .* p.grad
            end
        end
        @test all(p -> all(isfinite, Array(p.data)), get_params(Gt))
    end
end
