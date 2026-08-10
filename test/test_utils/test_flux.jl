using InvertibleNetworks, Flux, Test, LinearAlgebra

# Define network
nx = 1
ny = 1
n_in = 2
n_hidden = 10
batchsize = 32

# net
AN = ActNorm(n_in; logdet = false)
C = CouplingLayerGlow(n_in, n_hidden; logdet = false, k1 = 1, k2 = 1, p1 = 0, p2 = 0)
pan, pc = deepcopy(get_params(AN)), deepcopy(get_params(C))
model = Chain(AN, C)

# dummy input & target
X = randn(Float32, nx, ny, n_in, batchsize)
Y = model(X)
X0 = rand(Float32, nx, ny, n_in, batchsize) .+ 1

# loss fn
loss(model, X, Y) = Flux.mse(Y, model(X))

# Explicit Flux model gradients and optimiser state
opt_state = Flux.setup(Descent(0.001), model)
params_before = deepcopy(Flux.trainables(model))

l, grads = Flux.withgradient(model) do current_model
    loss(current_model, X0, Y)
end

model_grads = Flux.trainables(grads[1])
@test length(model_grads) == length(Flux.trainables(model))
for (parameter, parameter_grad) in zip(Flux.trainables(model), model_grads)
    @test !isnothing(parameter_grad)
    @test size(parameter_grad) == size(parameter)
end

Flux.update!(opt_state, model, grads[1])
@test any(!isapprox(before, after) for (before, after) in zip(params_before, Flux.trainables(model)))

for i = 1:5
    li, loop_grads = Flux.withgradient(model) do current_model
        loss(current_model, X, Y)
    end

    @info "Loss: $li"
    @test li != l
    global l = li

    Flux.update!(opt_state, model, loop_grads[1])
end

###################################################################################################
# Parameter gradient values through the explicit Flux interface.
#
# The rrule for `parameter_data` maps the flat gradients produced by the hand-written
# backward passes onto the nested structural tangent that Flux's explicit optimiser
# interface expects. Check the values Flux actually sees for every Parameter, not just
# that a gradient of the right shape exists.

G = NetworkGlow(n_in, n_hidden, 2, 2; logdet = false)
XG = randn(Float32, 4, 4, n_in, batchsize)
YG = G(XG)                     # first pass initializes the ActNorm parameters
XG0 = randn(Float32, 4, 4, n_in, batchsize)

# Gradient via Flux: the structural tangent assembled by the rrule
InvertibleNetworks.reset!(InvertibleNetworks.GLOBAL_STATE_INVOPS)
tangent = Flux.gradient(net -> .5f0*norm(net(XG0) - YG)^2, G)[1]

# The same gradient via the hand-written backward pass
InvertibleNetworks.reset!(InvertibleNetworks.GLOBAL_STATE_INVOPS)
clear_grad!(G)
ZG = G.forward(XG0)
G.backward(ZG - YG, ZG)

# `Flux.trainables` walks the network and the tangent in the same structural order,
# which is *not* the order of `get_params` (layer matrices are traversed row-major by
# one and column-major by the other), so pair them up by array identity instead.
backward_grads = IdDict{Any,Any}(p.data => p.grad for p in get_params(G))
expected_grads = [backward_grads[data] for data in Flux.trainables(G)]
flux_grads = Flux.trainables(tangent)

@test length(flux_grads) == length(get_params(G))
@test !any(isnothing, expected_grads)
@test all(isapprox(flux_g, expected_g; rtol = 1f-5)
          for (flux_g, expected_g) in zip(flux_grads, expected_grads))

# ... and that Flux's optimiser actually applies them to every parameter
η = 1f-2
params_pre = deepcopy(Flux.trainables(G))
Flux.update!(Flux.setup(Descent(η), G), G, tangent)
@test all(isapprox(param_post, param_pre - η*expected_g; rtol = 1f-4)
          for (param_post, param_pre, expected_g) in zip(Flux.trainables(G), params_pre, expected_grads))
