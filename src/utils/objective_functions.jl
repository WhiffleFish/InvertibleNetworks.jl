# Objective functions
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export log_likelihood, ∇log_likelihood, Hlog_likelihood, gaussian_lognorm, mse, ∇mse, Hmse

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
