using InvertibleNetworks, LinearAlgebra, Test, Random

Random.seed!(11)

# Input
nx = 16
ny = 16
nz = 16
n_in = 2
n_hidden = 4
batchsize = 2
maxiter = 2

# Observed data
nxrec = 10
nyrec = 12
nt = 8
d = randn(Float32, nt, nxrec*nyrec, 1, batchsize)

# Modeling/imaging operator
J = 1 .+ rand(Float32, nt*nxrec*nyrec, nx*ny*nz)

# Link function
Ψ(η) = identity(η)

# Unrolled loop
L = NetworkLoop3D(n_in, n_hidden, maxiter, Ψ; type="HINT")

# Initializations
η = 10*randn(Float32, nx, ny, nz, 1, batchsize)
s = 10*randn(Float32, nx, ny, nz, n_in-1, batchsize)

###################################################################################################

# Test invertibility
η_, s_ = L.forward(η, s, d, J)
ηInv, sInv = L.inverse(η_, s_, d, J)
@test isapprox(norm(ηInv - η)/norm(η), 0f0, atol=1e-5)
@test isapprox(norm(sInv - s)/norm(s), 0f0, atol=1e-5)

# Test invertibility
η_, s_ = L.forward(η, s, d, J)
ηInv, sInv = L.backward(0f0.*η_, 0f0.*s_, η_, s_, d, J)[3:4]
@test isapprox(norm(ηInv - η)/norm(η), 0f0, atol=1e-5)
@test isapprox(norm(sInv - s)/norm(sInv), 0f0, atol=1e-5)

η_, s_ = L.inverse(η, s, d, J)
ηInv, sInv = L.forward(η_, s_, d, J)
@test isapprox(norm(ηInv - η)/norm(η), 0f0, atol=1e-5)
@test isapprox(norm(sInv - s)/norm(sInv), 0f0, atol=1e-5)

###################################################################################################

# Initializations
η = randn(Float32, nx, ny, nz, 1, batchsize)
s = randn(Float32, nx, ny, nz, n_in-1, batchsize)
η0 = randn(Float32, nx, ny, nz, 1, batchsize)
s0 = randn(Float32, nx, ny, nz, n_in-1, batchsize)
Δη = η - η0
Δs = s - s0

# Observed data
η_, s_ = L.forward(η, s, d, J)   # only need η

function loss(L, η0, s0, d, η, s)
    η_, s_ = L.forward(η0, s0, d, J)  # reshape
    Δη = η_ - η
    Δs = s_ - s    # no "observed" s, so Δs=0
    f = .5f0*norm(Δη)^2 + .5f0*norm(Δs)^2
    Δη_, Δs_ = L.backward(Δη, Δs, η_, s_, d, J)[1:2]
    return f, Δη_, Δs_, L.L[1].C.v1.grad, L.L[1].CL[1].RB.W1.grad
end

# Gradient test for input
f0, gη, gs = loss(L, η0, s0, d, η_, s_)[1:3]
h = 0.1f0
maxiter = 6
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test loop unrolling\n")
for j=1:maxiter
    f = loss(L, η0 + h*Δη, s0 + h*Δs, d, η_, s_)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(Δη, gη) - h*dot(Δs, gs))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)


# Gradient test for weights
L0 = NetworkLoop3D(n_in, n_hidden, maxiter, Ψ; type="HINT")
L_ini = deepcopy(L0)
dv = L.L[1].C.v1.data - L0.L[1].C.v1.data   # just test for 2 parameters
dW = L.L[1].CL[1].RB.W1.data - L0.L[1].CL[1].RB.W1.data
f0, gη, gs, gv, gW = loss(L0, η, s, d, η_, s_)
h = 0.05f0
maxiter = 5
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test loop unrolling\n")
for j=1:maxiter
    L0.L[1].C.v1.data = L_ini.L[1].C.v1.data + h*dv
    L0.L[1].CL[1].RB.W1.data = L_ini.L[1].CL[1].RB.W1.data + h*dW
    f = loss(L0, η, s, d, η_, s_)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dv, gv) - h*dot(dW, gW))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)


###################################################################################################
# `type="additive"` is the documented default for NetworkLoop, but every test and example
# above uses type="HINT", so the additive path had no coverage at all.

@testset "additive coupling (default type)" begin
    # Self-contained inputs: the round-trip accuracy below is set by the Float32 Conv1x1
    # Householder inverse, so it is sensitive to the exact draw. The gradient tests above
    # mutate the module-level network/state, so draw fresh values from a pinned seed here
    # instead of inheriting whatever they left behind.
    Random.seed!(0xa1d1)
    d_add = randn(Float32, nt, nxrec*nyrec, 1, batchsize)
    J_add = 1 .+ rand(Float32, nt*nxrec*nyrec, nx*ny*nz)
    η_in = 10*randn(Float32, nx, ny, nz, 1, batchsize)
    s_in = 10*randn(Float32, nx, ny, nz, n_in-1, batchsize)

    L_add = NetworkLoop3D(n_in, n_hidden, maxiter, Ψ)
    @test eltype(L_add.L) <: CouplingLayerIRIM
    @test isconcretetype(eltype(L_add.L))

    η_add, s_add = L_add.forward(η_in, s_in, d_add, J_add)
    η_inv, s_inv = L_add.inverse(η_add, s_add, d_add, J_add)
    @test isapprox(norm(η_inv - η_in)/norm(η_in), 0f0, atol=1e-5)
    @test isapprox(norm(s_inv - s_in)/norm(s_in), 0f0, atol=1e-5)

    # The additive path keeps CouplingLayerIRIM's own ReLU default for its residual block,
    # and `rb_activation` is the knob that reaches it.
    @test L_add.L[1].RB.activation === ReLUlayer()
    L_sig = NetworkLoop3D(n_in, n_hidden, maxiter, Ψ; rb_activation=SigmoidLayer())
    @test L_sig.L[1].RB.activation === SigmoidLayer()

    @test_throws ArgumentError NetworkLoop3D(n_in, n_hidden, maxiter, Ψ; type="nonsense")
end
