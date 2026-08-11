# Residual block from Putzky and Welling (2019): https://arxiv.org/abs/1911.10914
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export FluxBlock

"""
    FB = FluxBlock(model::Chain)

 Create a (non-invertible) neural network block from a Flux network.

 *Input*: 

 - `model`: Flux neural network of type `Chain`

 *Output*:
 
 - `FB`: residual block layer

 *Usage:*

 - Forward mode: `Y = FB.forward(X)`

 - Backward mode: `ΔX = FB.backward(ΔY, X)`

 *Trainable parameters:*

 - Network parameters given by `Flux.trainables(model)`

 See also:  [`Chain`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
mutable struct FluxBlock{M<:Chain} <: NeuralNetLayer
    model::M
    params::Array{Parameter, 1}
end

Flux.@layer FluxBlock trainable=(params,)

block_forward(X, FB::FluxBlock) = forward(X, FB)

#######################################################################################################################
# Constructor

function FluxBlock(model::Chain)

    # Collect Flux parameters
    model_params = Flux.trainables(model)
    nparam = length(model_params)
    params = Array{Parameter}(undef, nparam)

    # Create InvertibleNetworks parameter
    for j=1:nparam
        params[j] = Parameter(model_params[j])
    end
    return FluxBlock(model, params)
end


#######################################################################################################################
# Functions

# Forward
forward(X::AbstractArray{T, N}, FB::FluxBlock) where {T, N} = FB.model(X)


# Gradients of a Flux model come back as a plain (named) tuple, which is walked with
# the default functor walk rather than the model's `trainable`. Pairing the two flat
# lists positionally would desynchronize for any layer whose `trainable` drops a
# numeric field, so look each gradient up by the key path of its parameter instead.
function model_gradients(model, model_grad)
    return [Flux.Functors.getkeypath(model_grad, path) for (path, _) in Flux.trainables(model; path=true)]
end


# Backward 2D
function backward(ΔY::AbstractArray{T, N}, X::AbstractArray{T, N}, FB::FluxBlock; set_grad::Bool=true) where {T, N}
    
    # Differentiate explicitly with respect to both the Flux model and input.
    _, back = Flux.pullback((model, input) -> model(input), FB.model, X)
    model_grad, ΔX = back(ΔY)
    param_grads = model_gradients(FB.model, model_grad)

    # Set gradients
    if set_grad
        for j=1:length(FB.params)
            FB.params[j].grad = param_grads[j]
        end
    else
        Δθ = Array{Parameter, 1}(undef, length(FB.params))
        for j=1:length(FB.params)
            Δθ[j] = Parameter(param_grads[j])
        end
    end

    set_grad ? (return ΔX) : (ΔX, Δθ)

end


## Jacobian utilities

function jacobian(::AbstractArray{T, N}, ::AbstractVector{<:Parameter}, ::AbstractArray{T, N}, ::FluxBlock) where {T, N}
    throw(ArgumentError("Jacobian for Flux block not yet implemented"))
end

function adjointJacobian(ΔY::AbstractArray{T, N}, X::AbstractArray{T, N}, FB::FluxBlock) where {T, N}
    return backward(ΔY, X, FB; set_grad=false)
end


## Other utils

# Clear gradients
function clear_grad!(FB::FluxBlock)
    nparams = length(FB.params)
    for j=1:nparams
        FB.params[j].grad = nothing
    end
end

"""
    P = get_params(NL::FluxBlock)

 Returns directly the Flux params array
"""
function get_params(FB::FluxBlock)
    return FB.params
end

function set_params!(FB::FluxBlock, θ::AbstractVector{<:Parameter})
    model_params = Flux.trainables(FB.model)
    nparams = length(model_params)
    for j=1:nparams
        model_params[j] .= θ[j].data
        FB.params[j].data = θ[j].data
        FB.params[j].grad = θ[j].grad
    end
end
