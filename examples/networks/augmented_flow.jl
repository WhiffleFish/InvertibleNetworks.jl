# Augmented normalizing flow on a four-mode 2D mixture.
#
# A plain flow has to map a unimodal Gaussian onto four separated modes with a homeomorphism,
# which forces it to stretch the map thin between them. Padding the input with auxiliary
# latent variables lets the modes be pulled apart along the extra axes instead, at the cost
# of a likelihood that is now a lower bound rather than exact.
#
# Huang, Dinh and Courville (2020): https://arxiv.org/abs/2002.07101
# Chen et al. (2020): https://arxiv.org/abs/2002.09741

using InvertibleNetworks, Flux, Random, Statistics, LinearAlgebra

Random.seed!(1234)

centers = [(-3f0,-3f0), (3f0,-3f0), (-3f0,3f0), (3f0,3f0)]

function sample_data(n)
    X = zeros(Float32, 1, 1, 2, n)
    for i = 1:n
        cx, cy = centers[rand(1:4)]
        X[1,1,1,i] = cx + 0.35f0*randn(Float32)
        X[1,1,2,i] = cy + 0.35f0*randn(Float32)
    end
    return X
end

# A Glow-style stack on `nc` channels. 1x1 kernels because the data has no spatial extent.
function make_flow(nc, nsteps, nhidden)
    layers = InvertibleNetworks.Invertible[]
    for _ = 1:nsteps
        push!(layers, ActNorm(nc; logdet=true))
        push!(layers, CouplingLayerGlow(nc, nhidden; logdet=true, k1=1, k2=1, p1=0, p2=0))
    end
    return InvertibleChain(tuple(layers...))
end

function train!(net, X; niter=1500, batchsize=256, lr=1f-3)
    opt_state = Flux.setup(Adam(lr), net)
    for _ = 1:niter
        Xb = X[:, :, :, rand(1:size(X, 4), batchsize)]
        _, grads = Flux.withgradient(m -> -log_likelihood(Xb, m), net)
        Flux.update!(opt_state, net, grads[1])
    end
    return net
end

Xtrain = sample_data(4096)
Xtest = sample_data(2048)
nsteps, nhidden, naug = 8, 32, 2

# Baseline: an ordinary flow, exact likelihood.
Random.seed!(7)
plain = make_flow(2, nsteps, nhidden)
plain.forward(Xtrain[:, :, :, 1:256])          # initialize ActNorm
train!(plain, Xtrain)

# Augmented: the same stack, two channels wider, wrapped so that the extra channels are
# filled with noise. The first coupling layer transforms the auxiliary half conditioned on
# the data half, which is exactly the learned q(ε|x) of ANF/VFlow -- no separate inference
# network needed.
Random.seed!(7)
augmented = AugmentedFlow(make_flow(2 + naug, nsteps, nhidden), naug)
augmented.forward(Xtrain[:, :, :, 1:256])
train!(augmented, Xtrain)

# `log_likelihood` returns the exact value for the plain flow and the ELBO for the augmented
# one; `log_likelihood_importance` tightens the latter with more draws of ε.
println("plain flow      log p(x)     : ", log_likelihood(Xtest, plain; normalized=true))
println("augmented flow  ELBO         : ", log_likelihood(Xtest, augmented; normalized=true))
println("augmented flow  IW bound K=32: ",
        log_likelihood_importance(Xtest, augmented; nsamples=32, normalized=true))

# Sampling. The augmented network's `inverse` drops the auxiliary channels for you.
function mode_purity(net, n, nc)
    X = net.inverse(randn(Float32, 1, 1, nc, n))
    d = [minimum(hypot(X[1,1,1,i] - cx, X[1,1,2,i] - cy) for (cx, cy) in centers) for i = 1:n]
    return mean(d .< 1.5f0)
end

println("plain flow      sample purity: ", mode_purity(plain, 2000, 2))
println("augmented flow  sample purity: ", mode_purity(augmented, 2000, 2 + naug))

# Note: the score `inverse_and_log_likelihood` reports for an AugmentedFlow is *not* a lower
# bound -- on the sampling path ε comes from the model rather than from q, which biases the
# estimate upward. Use an unaugmented flow where an exact density of your own samples is
# what matters.
