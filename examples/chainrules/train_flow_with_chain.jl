# Train a normalizing flow built with InvertibleChain, using Flux's explicit optimiser
# interface. The chain composes the layers, accumulates their log-determinants, and is
# differentiated by Zygote through the hand-written backward passes -- there is no manual
# layer-by-layer forward/backward bookkeeping.

using InvertibleNetworks, Flux, LinearAlgebra, Random

Random.seed!(11)

nx = ny = 8
n_in = 4
n_hidden = 32
batchsize = 32

# Build the flow the same way you would build a Flux.Chain: layers in application order.
flow = InvertibleChain(
    ActNorm(n_in; logdet=true),
    CouplingLayerGlow(n_in, n_hidden; logdet=true),
    ActNorm(n_in; logdet=true),
    CouplingLayerGlow(n_in, n_hidden; logdet=true),
)

# Target: correlated Gaussian samples, so the flow has something to learn.
sample_data(m) = begin
    Z = randn(Float32, nx, ny, n_in, m)
    2f0 .* Z .+ 1f0
end

# The objective is the negative log-likelihood of the data under the flow. By change of
# variables that is `-(log_likelihood(Z) + logdet)`, which `log_likelihood(X, flow)` gives
# directly; it is differentiable and usable both inside and outside a gradient.
loss(flow, X) = -log_likelihood(X, flow)

# Equivalently, if you want the latent as well:
#     Z, logdet = flow(X)
#     -log_likelihood(Z) - logdet

X = sample_data(batchsize)
init!(flow, X)     # ActNorm takes its statistics from this batch; until then the flow has
                   # no map, and `flow.inverse` says so rather than guessing

-log_likelihood(X, flow; normalize=false)

@profview for _ in 1:1000;flow(X);end

opt_state = Flux.setup(Adam(1f-3), flow)

println("initial loss: ", loss(flow, X))
for iter = 1:200
    Xb = sample_data(batchsize)
    l, grads = Flux.withgradient(m -> loss(m, Xb), flow)
    Flux.update!(opt_state, flow, grads[1])
    mod(iter, 50) == 0 && println("iteration $iter: loss = $l")
end
println("final loss:   ", loss(flow, X))

# The chain is still an invertible network: sample by pushing latents back through it.
Z = randn(Float32, nx, ny, n_in, 4)
X_sampled = flow.inverse(Z)
println("\ngenerated sample size: ", size(X_sampled))

# ... and the round trip holds.
Z_test, _ = flow(X)
println("reconstruction error: ", norm(flow.inverse(Z_test) - X) / norm(X))
