# Objective functions
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export log_likelihood, log_likelihood_per_sample, ∇log_likelihood, Hlog_likelihood,
       gaussian_lognorm, mse, ∇mse, Hmse

###################################################################################################
# Mean squared error
"""
    f = mse(X, Y)

Mean squared error between arrays/tensor X and Y

See also: [`∇mse`](@ref)
"""
mse(X::AbstractArray{T, N}, Y::AbstractArray{T, N}) where {T, N} = T(.5/size(X, N))*norm(X - Y, 2)^2


"""
    ∇f = ∇mse(X, Y)

Gradient of the MSE loss with respect to input tensors X and Y.

See also: [`mse`](@ref)
"""
∇mse(X::AbstractArray{T, N}, Y::AbstractArray{T, N}) where {T, N} = T(1/size(X, N))*(X - Y)


"""
    Hf = Hmse(X, Y)

Hessian of the MSE loss with respect to input tensors X.

See also: [`mse`](@ref)
"""
function Hmse(X::AbstractArray{T, N}, ::AbstractArray{T, N}) where {T, N}
    return InvertibleNetworkLinearOperator{Array{T, N},Array{T, N}}(
        ΔX -> T(1/size(X, N))*ΔX,
        ΔX -> T(1/size(X, N))*ΔX)
end


###################################################################################################
# Log-likelihood

"""
    f = log_likelihood(X; μ=T(0), σ=T(1), normalized=false)

Log-likelihood of X for a Gaussian distribution with given mean μ and variance
σ. All elements of X are assumed to be iid. The value is averaged over the batch
(the trailing dimension of X).

By default the Gaussian normalizing constant is dropped, so the result is a log-density
up to an additive constant -- enough for optimization, since the constant does not depend
on the model. Pass `normalized=true` to include `-d/2*log(2πσ²)`, where `d` is the number
of elements per sample, and get a calibrated log-density.

See also: [`∇log_likelihood`](@ref), [`gaussian_lognorm`](@ref)
"""
log_likelihood(X::AbstractArray{T, N}; μ=T(0), σ=T(1), normalized::Bool=false) where {T, N} =
    T(1/size(X, N))*sum(-T(.5)*((X .- μ)/σ).^2) + (normalized ? gaussian_lognorm(X, σ) : zero(T))


"""
    f = log_likelihood_per_sample(X; μ=T(0), σ=T(1), normalized=false)

Gaussian log-likelihood of each sample in the batched array `X`, returned as a vector of
length `size(X, N)`. This is the unaggregated form of [`log_likelihood`](@ref):
`sum(log_likelihood_per_sample(X))/size(X, N) == log_likelihood(X)`.

See also: [`log_likelihood`](@ref)
"""
function log_likelihood_per_sample(X::AbstractArray{T, N}; μ=T(0), σ=T(1),
                                  normalized::Bool=false) where {T, N}
    f = per_sample_sum(-T(.5)*((X .- μ)/σ).^2)
    return normalized ? f .+ gaussian_lognorm(X, σ) : f
end


###################################################################################################
# Per-sample reductions
#
# Every log-determinant in this package is a sum over the elements of one sample. Reducing
# over the batch dimension as well -- which the layers do by default, to return the batch
# average -- throws away exactly the information a per-sample score needs, and the only way
# to get it back is to re-run the network one sample at a time. Stopping the reduction one
# dimension early costs nothing and keeps it.

"""
    v = per_sample_sum(A)

Sum `A` over every dimension except the trailing batch dimension, giving a vector with one
entry per sample.
"""
@inline function per_sample_sum(A::AbstractArray{T, N}) where {T, N}
    dims = ntuple(i -> i, Val(N-1))     # everything except the batch dimension
    return dropdims(sum(A; dims=dims); dims=dims)
end

"""
    v = logdet_per_sample(S)

Log-determinant contributed by an elementwise scaling `S`, kept per sample: the unaggregated
form of `sum(log.(abs.(S)))/size(S)[end]`.
"""
@inline logdet_per_sample(S::AbstractArray) = per_sample_sum(log.(abs.(S)))

# A log-determinant that is the same for every sample (`ActNorm`, `AffineLayer`), spread
# over the batch so that it composes with the per-sample values of the other layers.
@inline constant_per_sample(X::AbstractArray{T, N}, v) where {T, N} =
    fill!(similar(X, T, size(X, N)), v)

# Weighting of the log-determinant in a backward pass.
#
# The hand-written `backward` of every layer computes the gradient of an objective whose
# log-determinant term is the batch-averaged `-logdet` with unit weight; in per-sample terms
# that is a weight of `-1/batchsize` on each sample. Passing an explicit weight vector `w`
# instead gives the gradient of `sum(w .* logdet_per_sample)`, which is what a loss with
# per-example weights needs and what no amount of reweighting a batch-averaged pass can
# recover. `nothing` keeps the historical behaviour exactly.

# Gradient contribution to an elementwise scaling `S` (coupling layers).
@inline logdet_scale_grad(S::AbstractArray{T, N}, ::Nothing) where {T, N} =
    -one(T)/size(S, N) ./ S
@inline logdet_scale_grad(S::AbstractArray{T, N}, w::AbstractVector) where {T, N} =
    reshape(w, ntuple(_ -> 1, Val(N-1))..., :) ./ S

# Weight applied to a raw log-determinant gradient array whose trailing dimension is the
# batch (elementwise bijectors).
@inline apply_logdet_weight(G::AbstractArray{T, N}, ::Nothing) where {T, N} =
    -G/size(G, N)
@inline apply_logdet_weight(G::AbstractArray{T, N}, w::AbstractVector) where {T, N} =
    reshape(w, ntuple(_ -> 1, Val(N-1))..., :) .* G

# Total weight seen by a log-determinant that is the same for every sample (`ActNorm`,
# `AffineLayer`): the per-sample weights simply add up.
@inline logdet_total_weight(::Nothing) = -1
@inline logdet_total_weight(w::AbstractVector) = sum(w)

# The per-sample weights are only defined for the gradient-accumulating pass; the
# Jacobian/Hessian interface (`set_grad=false`) reports the unweighted term separately.
@inline check_logdet_weight(::Nothing, ::Bool) = nothing
@inline check_logdet_weight(::AbstractVector, set_grad::Bool) = set_grad ? nothing :
    throw(ArgumentError("logdet_weight is only supported with set_grad=true"))

# Log-determinant modes accepted by the `logdet` keyword of `forward`/`inverse`:
#   `false`   -- do not compute it
#   `true`    -- the batch-averaged scalar the layers have always returned
#   `:sample` -- the per-sample vector that scalar is the mean of
@inline logdet_mode(logdet::Bool) = Val(logdet)
# `nothing` means "whatever the layer was built with"; resolved by dispatch rather than a
# branch so that inference does not have to prove the other branch unreachable.
@inline logdet_mode(::Nothing, default::Val) = default
@inline logdet_mode(logdet, ::Val) = logdet_mode(logdet)
@inline function logdet_mode(logdet::Symbol)
    logdet === :sample || throw(ArgumentError(
        "logdet must be `true`, `false` or `:sample`, got :$logdet"))
    return Val(:sample)
end


"""
    c = gaussian_lognorm(X, σ)

Per-sample Gaussian normalizing constant `-d/2*log(2πσ²)`, where `d = length(X) ÷ size(X, N)`
is the number of elements in one sample of the batched array `X`.

Since [`log_likelihood`](@ref) is averaged over the batch and this constant is the same for
every sample, it is added once rather than averaged.

See also: [`log_likelihood`](@ref)
"""
function gaussian_lognorm(X::AbstractArray{T, N}, σ) where {T, N}
    d = length(X) ÷ size(X, N)
    return T(-d/2 * log(2*π*σ^2))
end


"""
    ∇f = ∇log_likelihood(X; μ=T(0), σ=T(1))

Gradient of the Gaussian log-likelihood function with respect to the input
tensor X.

See also: [`log_likelihood`](@ref)
"""
∇log_likelihood(X::AbstractArray{T, N}; μ=T(0), σ=T(1)) where {T, N} = T(-1/size(X, N))*(X .- μ)/σ^2


"""
    Hf = Hlog_likelihood(X; μ=T(0), σ=T(1))

Hessian of the Gaussian log-likelihood function with respect to the input
tensor X.

See also: [`log_likelihood`](@ref)
"""
function Hlog_likelihood(X::AbstractArray{T, N}; μ=T(0), σ=T(1)) where {T, N}
    return InvertibleNetworkLinearOperator{AbstractArray{T, N},AbstractArray{T, N}}(
        ΔX -> -T(1/size(X, N))*ΔX/σ^2,
        ΔX -> -T(1/size(X, N))*ΔX/σ^2)
end
