# Test residual block
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using LinearAlgebra, InvertibleNetworks, Test, Flux

# Input
nx = 28
ny = 28
n_in = 4
n_hidden = 8
batchsize = 2

# Flux networks
model = Chain(
    Conv((3,3), n_in => n_hidden; pad=1),
    Conv((3,3), n_hidden => n_in; pad=1)
)

model0 = Chain(
    Conv((3,3), n_in => n_hidden; pad=1),
    Conv((3,3), n_hidden => n_in; pad=1)
)

# Input
X = glorot_uniform(nx, ny, n_in, batchsize)
X0 = glorot_uniform(nx, ny, n_in, batchsize)
dX = X - X0

# Flux blocks
RB = FluxBlock(model)
RB0 = FluxBlock(model0)

# Observed data
Y = RB.forward(X)

function loss(RB, X, Y)
    Y_ = RB.forward(X)
    ΔY = Y_ - Y
    f = .5f0*norm(ΔY)^2
    ΔX = RB.backward(ΔY, X)
    return f, ΔX, RB.params[1].grad, RB.params[2].grad
end


###################################################################################################
# Gradient tests

# Gradient test w.r.t. input
f0, ΔX = loss(RB, X0, Y)[1:2]
h = 0.1f0
maxiter = 5
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test convolutions\n")
for j=1:maxiter
    f = loss(RB, X0 + h*dX, Y)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, ΔX))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)


# Gradient test for weights
RB_ini = deepcopy(RB0)
dW1 = RB.params[1].data - RB0.params[1].data
dW2 = RB.params[2].data - RB0.params[2].data

f0, ΔX, ΔW1, ΔW2 = loss(RB0, X, Y)
h = 0.1f0
maxiter = 5
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test convolutions\n")
for j=1:maxiter
    RB0.params[1].data[:] = RB_ini.params[1].data + h*dW1
    RB0.params[2].data[:] = RB_ini.params[2].data + h*dW2
    f = loss(RB0, X, Y)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dW1, ΔW1) - h*dot(dW2, ΔW2))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)


###################################################################################################
# Gradients must stay paired with their parameter for layers that restrict `trainable`.
#
# A Flux gradient comes back as a plain named tuple, which is walked with the default
# functor walk rather than the layer's `trainable`. Pairing the trainable arrays and the
# gradient leaves positionally desynchronizes as soon as `trainable` drops a numeric
# field, so `backward` looks each gradient up by its parameter's key path.

struct ScaleShift
    scale
    shift
end

(SS::ScaleShift)(X) = SS.scale .* X .+ SS.shift

# `shift` is declared second on purpose: pairing the two flat lists positionally would
# hand the gradient of `scale` to `shift`, which this test catches.
Flux.@layer ScaleShift trainable=(shift,)

n_feat = 4
batch_partial = 3
SS = ScaleShift(randn(Float32, n_feat), randn(Float32, n_feat))
FB_partial = FluxBlock(Chain(SS))

# Only `shift` is trainable, but the gradient tuple carries both `scale` and `shift`
@test length(FB_partial.params) == 1
@test FB_partial.params[1].data === SS.shift

X_partial = randn(Float32, n_feat, batch_partial)
FB_partial.backward(ones(Float32, n_feat, batch_partial), X_partial)

# For f = sum(scale .* X .+ shift): df/dshift is the batch size, while df/dscale would be
# the row sums of X. Getting the latter here would mean the gradients came back misaligned.
@test FB_partial.params[1].grad ≈ fill(Float32(batch_partial), n_feat)
