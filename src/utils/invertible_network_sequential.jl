# Invertible network obtained from stringing invertible networks together
# Author: Gabrio Rizzuti, grizzuti3@gatech.edu
# Date: September 2020

export ComposedInvertibleNetwork, Composition
import Base.length, Base.∘

struct ComposedInvertibleNetwork{L<:Tuple,D<:Tuple,P<:Tuple} <: InvertibleNetwork
    layers::L
    logdet_array::D
    logdet::Bool
    npars::P
end

Flux.@layer ComposedInvertibleNetwork


## Constructors

function Composition(layer...)

    depth = length(layer)
    layers = reverse(layer)
    logdet_array = ntuple(i -> hasproperty(layers[i], :logdet) && layers[i].logdet, depth)
    logdet = any(logdet_array)
    npars = ntuple(i -> length(get_params(layers[i])), depth)

    return ComposedInvertibleNetwork(layers, logdet_array, logdet, npars)

end


## Composition utilities

function ∘(net1::ComposedInvertibleNetwork, net2::ComposedInvertibleNetwork)
    return Composition(net1.layers[end:-1:1]..., net2.layers[end:-1:1]...)
end

function ∘(net1::Union{NeuralNetLayer, InvertibleNetwork}, net2::Union{NeuralNetLayer, InvertibleNetwork})
    return Composition(net1, net2)
end

function ∘(net1::Union{NeuralNetLayer, InvertibleNetwork}, net2::ComposedInvertibleNetwork)
    return Composition(net1, net2.layers[end:-1:1]...)
end

function ∘(net1::ComposedInvertibleNetwork, net2::Union{NeuralNetLayer, InvertibleNetwork})
    return Composition(net1.layers[end:-1:1]..., net2)
end

function length(N::ComposedInvertibleNetwork)
    return length(N.layers)
end


## Forward/inverse/backward

function forward(X::AbstractArray{T, N1}, N::ComposedInvertibleNetwork) where {T, N1}
    # `zero(T)`, not `0`: a literal Int accumulator widens the return type to
    # Union{T,Int}. Note this loop is still not inferrable overall, because `N.layers[i]`
    # indexes a heterogeneous tuple with a runtime index -- see `InvertibleChain` for the
    # type-stable, recursively-unrolled equivalent.
    N.logdet && (logdet = zero(T))
    for i = 1:length(N)
        if N.logdet_array[i]        
            X, logdet_ = N.layers[i].forward(X)
            logdet += logdet_
        else
            X = N.layers[i].forward(X)
        end
    end
    N.logdet ? (return X, logdet) : (return X)
end

function inverse(Y::AbstractArray{T, N1}, N::ComposedInvertibleNetwork) where {T, N1}
    for i = length(N):-1:1
        Y = N.layers[i].inverse(Y)
    end
    return Y
end

function backward(ΔY::AbstractArray{T, N1}, Y::AbstractArray{T, N1}, N::ComposedInvertibleNetwork; set_grad::Bool = true) where {T, N1}
    if ~set_grad
        Δθ = Array{Parameter, 1}(undef, 0)
        N.logdet && (∇logdet = Array{Parameter, 1}(undef, 0))
    end
    for i = length(N):-1:1
        if set_grad
            ΔY, Y = N.layers[i].backward(ΔY, Y)
        else
            if N.logdet_array[i]
                ΔY, Δθi, Y, ∇logdet_i = N.layers[i].backward(ΔY, Y; set_grad=set_grad)
                ∇logdet = cat(∇logdet_i, ∇logdet; dims=1)
            else
                ΔY, Δθi, Y = N.layers[i].backward(ΔY, Y; set_grad=set_grad)
            end
            Δθ = cat(Δθi, Δθ; dims=1)
        end
    end
    if set_grad
        return ΔY, Y
    else
        N.logdet ? (return ΔY, Δθ, Y, ∇logdet) : (return ΔY, Δθ, Y)
    end
end


## Jacobian-related utilities

function jacobian(ΔX::AbstractArray{T, N1}, Δθ::AbstractVector{<:Parameter}, X::AbstractArray{T, N1}, N::ComposedInvertibleNetwork) where {T, N1}
    N.logdet && (l = 0; GNΔθ = Array{Parameter, 1}(undef, 0))
    idx_pars = 0
    for i = 1:length(N)
        npars_i = N.npars[i]
        Δθi = Δθ[idx_pars+1:idx_pars+npars_i]
        if N.logdet_array[i]
            ΔX, X, li, GNΔθi = N.layers[i].jacobian(ΔX, Δθi, X)
            l += li
            GNΔθ = cat(GNΔθ, GNΔθi; dims=1)
        else
            ΔX, X = N.layers[i].jacobian(ΔX, Δθi, X)
        end
        idx_pars += npars_i
    end
    N.logdet ? (return ΔX, X, l, GNΔθ) : (return ΔX, X)
end

function adjointJacobian(ΔY::AbstractArray{T, N1}, Y::AbstractArray{T, N1}, N::ComposedInvertibleNetwork) where {T, N1}
    return backward(ΔY, Y, N; set_grad = false)
end
