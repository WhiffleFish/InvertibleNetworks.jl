# Train networks with flux. Only guaranteed to work with logdet=false for now. 
# So you can train them as invertible networks like this, not as normalizing flows. 
using InvertibleNetworks, Flux

# Glow Network
model = NetworkGlow(2, 32, 2, 5; logdet=false)

# dummy input & target
X = randn(Float32, 16, 16, 2, 2) 
Y = 2 .* X .+ 1

# loss fn
loss(model, X, Y) = Flux.mse(Y, model(X))

# Run one forward pass before setting up the optimizer: ActNorm initializes its
# parameters during the first pass, and Flux.setup silently skips any parameter whose
# data is still nothing (leaving those weights frozen for the whole training run).
model(X)

opt_state = Flux.setup(Adam(0.0001f0), model)

for i = 1:500
    l, grads = Flux.withgradient(model) do current_model
        loss(current_model, X, Y)
    end
    @show l
    Flux.update!(opt_state, model, grads[1])
end
