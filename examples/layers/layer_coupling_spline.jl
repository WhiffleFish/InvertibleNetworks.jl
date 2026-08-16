# Neural spline flow coupling layer from Durkan et al. (2019): https://arxiv.org/abs/1906.04032
#
# The affine `s .* x .+ t` of a Glow coupling layer is replaced by a monotonic spline whose
# knots the residual block predicts. It still inverts analytically in one pass, but a single
# layer can now bend the map differently in each of its K bins -- enough to fit a multi-modal
# conditional that an affine coupling would need many layers to build up.

using InvertibleNetworks, LinearAlgebra, Test

# Input
nx = 32
ny = 32
k = 8
n_hidden = 16
batchsize = 4

X  = randn(Float32, nx, ny, k, batchsize)
X0 = randn(Float32, nx, ny, k, batchsize)

# Monotonic rational-quadratic splines with 8 bins over [-3, 3], the paper's default.
L = CouplingLayerSpline(k, n_hidden; spline=:rqs, nbins=8, bound=3f0, logdet=true)

# Zero-initialized by default, so the spline starts as the identity -- a deep stack of these
# trains stably from the first step. The 1x1 convolution in front still rotates, as in Glow, so
# it is the log-determinant rather than the output that gives the identity away.
@test isapprox(L.forward(X)[2], 0f0; atol=1f-3)   # a sum over 32x32x4 knots, in Float32
@test isapprox(SplineLayer(k; logdet=true).forward(X)[1], X; atol=1f-4)   # no mixing here

# Give the conditioner some weight so the layer actually bends.
L = CouplingLayerSpline(k, n_hidden; nbins=8, bound=3f0, logdet=true, zero_init=false)

# Forward + inverse: the inverse is exact and costs one conditioner pass, like the affine layer
Y, logdet = L.forward(X)
@test isapprox(norm(X - L.inverse(Y))/norm(X), 0f0; atol=1f-4)

# The log-determinant is available per sample, not just averaged over the batch
_, per_sample = L.forward(X; logdet=:sample)
@test length(per_sample) == batchsize
@test isapprox(sum(per_sample)/batchsize, logdet; rtol=1f-4)

# Backward
Y0 = L.forward(X0)[1]
ΔX, X0_ = L.backward(Y0 - Y, Y0)
@test isapprox(norm(X0_ - X0)/norm(X0), 0f0; atol=1f-2)


## Variants ---------------------------------------------------------------------------------

# Linear rational splines (Dolatabadi et al., 2020). Same monotone-spline idea, but the inverse
# is a linear solve rather than a quadratic root -- worth preferring when the flow is sampled
# from far more often than it is scored.
Llrs = CouplingLayerSpline(k, n_hidden; spline=:lrs, nbins=8, bound=3f0, logdet=true,
                           zero_init=false)
@test isapprox(norm(X - Llrs.inverse(Llrs.forward(X)[1]))/norm(X), 0f0; atol=1f-4)

# Circular splines (Rezende et al., 2020) for angle-valued channels. These act on the torus
# [-B, B), which a 1x1 convolution does not preserve, so they carry no mixing; alternate `swap`
# between layers instead so that every channel gets transformed.
B = 3f0
Xa = B .* (2f0 .* rand(Float32, nx, ny, k, batchsize) .- 1f0)   # angles, rescaled to [-B, B)
torus = InvertibleChain(
    CouplingLayerSpline(k, n_hidden; spline=:circular, nbins=8, bound=B, logdet=true,
                        zero_init=false),
    CouplingLayerSpline(k, n_hidden; spline=:circular, nbins=8, bound=B, logdet=true,
                        zero_init=false, swap=true))
Za = torus.forward(Xa)[1]
@test all(-B .<= Za .< B)                                        # still on the torus
@test isapprox(norm(Xa - torus.inverse(Za))/norm(Xa), 0f0; atol=1f-4)

# An elementwise spline, with free trainable knots instead of predicted ones. Useful as a
# strictly more flexible ActNorm, and as the per-channel warp the paper applies to the half of
# a coupling layer that would otherwise pass through untouched.
SL = SplineLayer(k; nbins=8, bound=3f0, logdet=true)
SL.θ.data .= 0.5f0 .* randn(Float32, size(SL.θ.data))
@test isapprox(norm(X - SL.inverse(SL.forward(X)[1]))/norm(X), 0f0; atol=1f-4)
