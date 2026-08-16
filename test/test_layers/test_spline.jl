# Neural spline flow layers: Durkan et al. (2019), https://arxiv.org/abs/1906.04032
# Circular splines: Rezende et al. (2020). Linear rational splines: Dolatabadi et al. (2020).

using InvertibleNetworks, LinearAlgebra, Test, Random, Flux

Random.seed!(2024)

const IN = InvertibleNetworks
const KINDS = (:rqs, :circular, :lrs)

function median_(x)
    s = sort(x)
    n = length(s)
    return isodd(n) ? s[(n+1)÷2] : (s[n÷2] + s[n÷2+1])/2
end

# Second-order Taylor test: `f` is a scalar loss of a perturbation size `h`, `g` the directional
# derivative predicted by the hand-written backward pass. The zeroth-order error must fall like
# `h` and the first-order error like `h²`, which it only does if `g` is right.
#
# Summarized by the median of the successive ratios rather than their mean: the first-order error
# is a difference of two nearly equal numbers, so one step of the sweep can land near a sign
# change and blow its ratio up without saying anything about the gradient.
function taylor_test(f, f0, g; h0=1f-1, hfactor=8f-1, maxiter=6)
    err1 = zeros(Float32, maxiter)
    err2 = zeros(Float32, maxiter)
    h = h0
    for j = 1:maxiter
        fh = f(h)
        err1[j] = abs(fh - f0)
        err2[j] = abs(fh - f0 - h*g)
        h *= hfactor
    end
    return median_(err1[1:end-1] ./ err1[2:end]), median_(err2[1:end-1] ./ err2[2:end])
end

const EXPECT1 = 1f0/8f-1
const EXPECT2 = 1f0/8f-1^2f0


###################################################################################################
@testset "Spline transform, kind = :$kind" for kind in KINDS

    K, B = 6, 3f0
    spec = SplineSpec(kind; nbins=K, bound=B)
    P = n_spline_params(spec)
    @test P == (kind === :rqs ? 3K-1 : kind === :circular ? 3K : 4K-1)

    # Working layout of the layers: value `(C, 1, N)`, parameters `(C, P, N)`, bin axis 2.
    C, NB = 3, 5
    θ = randn(Float32, C, P, NB)
    x = 2.5f0 .* randn(Float32, C, 1, NB)
    kn = IN.spline_knots(θ, spec, Val(2))

    y, lg = IN.spline_forward(x, kn, spec, Val(2))
    xr, lgr = IN.spline_inverse(y, kn, spec, Val(2))

    # A circular spline has no outside: it wraps its input into one period first.
    xref = kind === :circular ? mod.(x .+ B, 2B) .- B : x
    @test isapprox(norm(xr - xref)/norm(xref), 0f0; atol=1f-5)
    @test isapprox(lgr, lg; rtol=1f-4)

    # The bins tile the interval exactly, so the spline meets its tails without a seam.
    @test isapprox(sum(kn.binw; dims=2), fill(2B, C, 1, NB); rtol=1f-5)
    @test isapprox(sum(kn.binh; dims=2), fill(2B, C, 1, NB); rtol=1f-5)

    # log|dy/dx| against a central difference.
    h = 1f-3
    fd = log.(abs.((IN.spline_forward(x .+ h, kn, spec, Val(2))[1] .-
                    IN.spline_forward(x .- h, kn, spec, Val(2))[1]) ./ (2h)))
    @test isapprox(fd, lg; atol=1f-2)

    # Strictly increasing across one period, which is what makes the map invertible at all.
    grid = reshape(collect(range(-B + 1f-3, B - 1f-3; length=500)), 1, 1, :)
    kn1 = IN.spline_knots(repeat(θ[1:1, :, 1:1], 1, 1, 500), spec, Val(2))
    ys, _ = IN.spline_forward(Float32.(grid), kn1, spec, Val(2))
    @test all(diff(vec(ys)) .> 0)

    # `:rqs` and `:lrs` are the identity outside [-B, B]; `:circular` has no outside.
    if kind !== :circular
        far = reshape(Float32[-2B, 2B], 1, 1, :)
        knf = IN.spline_knots(repeat(θ[1:1, :, 1:1], 1, 1, 2), spec, Val(2))
        yf, lgf = IN.spline_forward(far, knf, spec, Val(2))
        @test yf == far
        @test all(iszero, lgf)
    end

    # All-zero parameters must be exactly the identity, which is what `identity_init` buys and
    # what makes a deep spline flow start as a no-op.
    kn0 = IN.spline_knots(zeros(Float32, C, P, NB), spec, Val(2))
    y0, lg0 = IN.spline_forward(x, kn0, spec, Val(2))
    @test isapprox(y0, xref; atol=1f-4)
    @test isapprox(lg0, zeros(Float32, C, 1, NB); atol=1f-4)
end


###################################################################################################
@testset "Spline VJP vs finite differences, kind = :$kind" for kind in KINDS

    spec = SplineSpec(kind; nbins=5, bound=3f0)
    P = n_spline_params(spec)
    C, NB = 2, 3
    θ  = 0.5f0 .* randn(Float32, C, P, NB)
    x  = 2.5f0 .* randn(Float32, C, 1, NB)
    Δy = randn(Float32, C, 1, NB)
    Δl = 0.37f0

    loss(xx, tt) = begin
        y, l = IN.spline_forward(xx, IN.spline_knots(tt, spec, Val(2)), spec, Val(2))
        sum(Δy .* y) + Δl*sum(l)
    end

    gx, gθ = IN.spline_vjp(Δy, Δl, x, IN.spline_knots(θ, spec, Val(2)), spec, Val(2))

    h = 1f-3
    central(f, A, i) = begin
        Ap = copy(A); Ap[i] += h
        Am = copy(A); Am[i] -= h
        (f(Ap) - f(Am))/(2h)
    end
    gx_fd = [central(a -> loss(a, θ), x, i) for i in eachindex(x)]
    gθ_fd = [central(t -> loss(x, t), θ, i) for i in eachindex(θ)]

    @test isapprox(vec(gx), gx_fd; rtol=1f-2)
    @test isapprox(vec(gθ), gθ_fd; rtol=1f-2)
end


###################################################################################################
@testset "CouplingLayerSpline, kind = :$kind" for kind in KINDS

    nx, ny, k, n_hidden, batchsize = 8, 8, 4, 4, 2

    # A circular layer maps the torus to itself, so its input has to be on the torus -- and stay
    # there under the Taylor test's perturbations, hence the margin. The other two are the
    # identity outside [-B, B] and take anything.
    B = 4f0
    draw() = kind === :circular ? 3f0 .* (2f0 .* rand(Float32, nx, ny, k, batchsize) .- 1f0) :
                                  2f0 .* randn(Float32, nx, ny, k, batchsize)
    X, X0 = draw(), draw()
    dX = X - X0

    L = CouplingLayerSpline(k, n_hidden; spline=kind, nbins=5, bound=B, logdet=true,
                            zero_init=false)

    # Circular layers refuse the 1x1 convolution, which would take them off the torus.
    if kind === :circular
        @test isnothing(L.C)
        @test_throws ArgumentError CouplingLayerSpline(k, n_hidden; spline=:circular, mix=true)
    else
        @test L.C isa Conv1x1
    end

    # Invertibility, both ways round.
    @test isapprox(norm(X - L.inverse(L.forward(X)[1]))/norm(X), 0f0; atol=1f-4)
    @test isapprox(norm(X - L.forward(L.inverse(X))[1])/norm(X), 0f0; atol=1f-4)

    # The inverse reports the negative of the forward log-determinant.
    Y, lgdet = L.forward(X)
    @test isapprox(L.inverse(Y; logdet=true)[2], -lgdet; rtol=1f-4)

    # Per-sample log-determinant averages to the scalar one.
    @test supports_per_sample_logdet(L)
    Ys, per_sample = L.forward(X; logdet=:sample)
    @test Ys ≈ Y
    @test length(per_sample) == batchsize
    @test isapprox(sum(per_sample)/batchsize, lgdet; rtol=1f-4)
    @test isapprox(L.inverse(Y; logdet=:sample)[2], -per_sample; rtol=1f-4)

    # A zero-initialized conditioner gives the identity spline, hence a zero log-determinant.
    Lid = CouplingLayerSpline(k, n_hidden; spline=kind, nbins=5, bound=B, logdet=true)
    @test isapprox(Lid.forward(X)[2], 0f0; atol=1f-4)

    # `swap` transforms the other half, which is how a stack without mixing reaches every
    # channel. It has to invert just as exactly.
    Ls = CouplingLayerSpline(k, n_hidden; spline=kind, nbins=5, bound=B, logdet=true,
                             zero_init=false, swap=true)
    @test isapprox(norm(X - Ls.inverse(Ls.forward(X)[1]))/norm(X), 0f0; atol=1f-4)

    ## Gradient w.r.t. the input
    function loss(L, X, Y)
        Y_, logdet = L.forward(X)
        f = mse(Y_, Y) - logdet
        ΔX = L.backward(∇mse(Y_, Y), Y_)[1]
        return f, ΔX
    end

    Yt = L.forward(X)[1]
    f0, ΔX = loss(L, X0, Yt)
    r1, r2 = taylor_test(h -> loss(L, X0 + h*dX, Yt)[1], f0, dot(dX, ΔX))
    @test isapprox(r1, EXPECT1; atol=1f0)
    @test isapprox(r2, EXPECT2; atol=1f0)

    ## Gradient w.r.t. the conditioner's weights
    clear_grad!(L)
    f0, = loss(L, X0, Yt)
    ΔW = copy(L.RB.W2.grad)
    dW = randn(Float32, size(L.RB.W2.data)); dW .*= norm(L.RB.W2.data)/norm(dW)
    W0 = copy(L.RB.W2.data)
    function lossW(h)
        L.RB.W2.data .= W0 .+ h .* dW
        f = loss(L, X0, Yt)[1]
        L.RB.W2.data .= W0
        return f
    end
    r1, r2 = taylor_test(lossW, f0, dot(dW, ΔW))
    @test isapprox(r1, EXPECT1; atol=1f0)
    @test isapprox(r2, EXPECT2; atol=1f0)
end


###################################################################################################
@testset "SplineLayer, kind = :$kind" for kind in KINDS

    nx, k, batchsize = 6, 3, 4

    # As for the coupling layer, a circular spline is only invertible on its own domain.
    B = 4f0
    draw() = kind === :circular ? 3f0 .* (2f0 .* rand(Float32, nx, k, batchsize) .- 1f0) :
                                  2f0 .* randn(Float32, nx, k, batchsize)
    X, X0 = draw(), draw()
    dX = X - X0

    L = SplineLayer(k; spline=kind, nbins=6, bound=B, logdet=true)

    # Zero-initialized, so it starts as the identity.
    @test isapprox(L.forward(X)[1], X; atol=1f-4)
    @test isapprox(L.forward(X)[2], 0f0; atol=1f-4)

    # A circular spline does wrap anything handed to it from outside the torus, which is a
    # projection and not something the layer claims to invert.
    if kind === :circular
        @test isapprox(L.forward(X .+ 2B)[1], X; atol=1f-3)
    end

    L.θ.data .= 0.6f0 .* randn(Float32, size(L.θ.data))

    @test isapprox(norm(X - L.inverse(L.forward(X)[1]))/norm(X), 0f0; atol=1f-4)
    Y, lgdet = L.forward(X)
    @test isapprox(L.inverse(Y; logdet=true)[2], -lgdet; rtol=1f-4)

    @test supports_per_sample_logdet(L)
    _, per_sample = L.forward(X; logdet=:sample)
    @test length(per_sample) == batchsize
    @test isapprox(sum(per_sample)/batchsize, lgdet; rtol=1f-4)

    function loss(L, X, Y)
        Y_, logdet = L.forward(X)
        f = mse(Y_, Y) - logdet
        ΔX = L.backward(∇mse(Y_, Y), Y_)[1]
        return f, ΔX
    end

    Yt = L.forward(X)[1]
    f0, ΔX = loss(L, X0, Yt)
    r1, r2 = taylor_test(h -> loss(L, X0 + h*dX, Yt)[1], f0, dot(dX, ΔX))
    @test isapprox(r1, EXPECT1; atol=1f0)
    @test isapprox(r2, EXPECT2; atol=1f0)

    ## Gradient w.r.t. the layer's own knots
    clear_grad!(L)
    f0, = loss(L, X0, Yt)
    Δθ = copy(L.θ.grad)
    dθ = randn(Float32, size(L.θ.data))
    θ0 = copy(L.θ.data)
    function lossθ(h)
        L.θ.data .= θ0 .+ h .* dθ
        f = loss(L, X0, Yt)[1]
        L.θ.data .= θ0
        return f
    end
    r1, r2 = taylor_test(lossθ, f0, dot(dθ, Δθ))
    @test isapprox(r1, EXPECT1; atol=1f0)
    @test isapprox(r2, EXPECT2; atol=1f0)
end


###################################################################################################
@testset "Per-sample log-determinant weighting" begin

    nx, k, batchsize = 5, 4, 3
    X = 2f0 .* randn(Float32, nx, k, batchsize)
    L = CouplingLayerSpline(k, 4; nbins=5, bound=4f0, logdet=true, zero_init=false, ndims=1)

    Y = L.forward(X)[1]
    ΔY = zeros(Float32, size(Y))
    w = Float32[0.3, -1.1, 0.7]

    # `logdet_weight = w` asks for the gradient of `sum(w .* logdet_per_sample)`; the default
    # is the gradient of `-logdet`, i.e. a uniform weight of `-1/batchsize`.
    clear_grad!(L); L.backward(ΔY, Y; logdet_weight=w)
    gw = copy(L.RB.W2.grad)
    clear_grad!(L); L.backward(ΔY, Y; logdet_weight=fill(-1f0/batchsize, batchsize))
    gu = copy(L.RB.W2.grad)
    clear_grad!(L); L.backward(ΔY, Y)
    gd = copy(L.RB.W2.grad)

    @test isapprox(gu, gd; rtol=1f-4)
    @test !isapprox(gw, gd; rtol=1f-2)

    # Weighting is linear in `w`, so scaling the weights scales the gradient.
    clear_grad!(L); L.backward(ΔY, Y; logdet_weight=2 .* w)
    @test isapprox(copy(L.RB.W2.grad), 2 .* gw; rtol=1f-4)
end


###################################################################################################
@testset "Precision conversion, kind = :$kind" for kind in KINDS

    # `Flux.f64` walks the layer and rebuilds every child. `SplineSpec` carries its kind in a
    # type parameter that the field values alone cannot recover, so it has to be a leaf.
    L = SplineLayer(3; spline=kind, nbins=6, bound=3f0, logdet=true)
    L.θ.data .= 0.5f0 .* randn(Float32, size(L.θ.data))
    L64 = Flux.f64(L)
    @test L64.spline === L.spline

    X = 2 .* rand(Float64, 4, 3, 5) .- 1
    Y, lgdet = L64.forward(X)
    @test eltype(Y) == Float64
    @test lgdet isa Float64
    @test isapprox(norm(X - L64.inverse(Y))/norm(X), 0f0; atol=1e-12)

    CL64 = Flux.f64(CouplingLayerSpline(4, 8; spline=kind, nbins=6, bound=3f0, ndims=1,
                                        logdet=true, zero_init=false))
    Xc = 2 .* rand(Float64, 6, 4, 3) .- 1
    @test isapprox(norm(Xc - CL64.inverse(CL64.forward(Xc)[1]))/norm(Xc), 0f0; atol=1e-12)
end


###################################################################################################
@testset "Spline flow in an InvertibleChain" begin

    nx, k, batchsize, n_hidden = 6, 4, 8, 8
    X = 0.7f0 .* randn(Float32, nx, k, batchsize)

    flow = InvertibleChain(ActNorm(k; logdet=true),
                           CouplingLayerSpline(k, n_hidden; nbins=6, bound=5f0, logdet=true,
                                               zero_init=false, ndims=1),
                           SplineLayer(k; nbins=6, bound=5f0, logdet=true),
                           CouplingLayerSpline(k, n_hidden; spline=:lrs, nbins=6, bound=5f0,
                                               logdet=true, zero_init=false, ndims=1))
    init!(flow, X)

    Z, lgdet = flow.forward(X)
    @test isapprox(norm(X - flow.inverse(Z))/norm(X), 0f0; atol=1f-4)
    @test isapprox(flow.inverse(Z; logdet=true)[2], -lgdet; rtol=1f-4)

    @test supports_per_sample_logdet(flow)
    _, per_sample = flow.forward(X; logdet=:sample)
    @test length(per_sample) == batchsize
    @test isapprox(sum(per_sample)/batchsize, lgdet; rtol=1f-4)

    # A per-sample score and its batch average must agree with the scalar path.
    scores = log_likelihood_per_sample(X, flow)
    @test length(scores) == batchsize
    @test isapprox(sum(scores)/batchsize, log_likelihood(Z) + lgdet; rtol=1f-3)

    # Zygote must reproduce the hand-written gradient, since that is how the chain is trained.
    l, grads = Flux.withgradient(m -> begin
        Zm, ld = m(X)
        -log_likelihood(Zm) - ld
    end, flow)
    @test isfinite(l)
    # Zygote returns a structural tangent; only `Parameter.data` is trainable.
    g_cl = grads[1].layers[2].RB.W2.data
    @test g_cl !== nothing && all(isfinite, g_cl)

    clear_grad!(flow)
    Zm = flow.forward(X)[1]
    flow.backward(-∇log_likelihood(Zm), Zm)
    @test isapprox(flow[2].RB.W2.grad, g_cl; rtol=1f-3)
end
