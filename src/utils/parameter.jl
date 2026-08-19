# Parameter of neural network
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export Parameter, get_params, get_grads, set_params!, par2vec, vec2par
export OUT_INIT_SCALE

mutable struct Parameter{D}
    data::D
    grad::Union{Nothing,D}
end

Parameter(::Nothing) = Parameter{Nothing}(nothing, nothing)

convert_param!(::Type{T}, ::Parameter{D}) where {T,D<:AbstractArray{T}} = nothing
function convert_param!(::Type{T}, p::Parameter) where T
    throw(ArgumentError("cannot change a Parameter's concrete storage type in place; " *
                        "convert the model with Flux.f32, Flux.f64, cpu, or gpu"))
end

"""
    Class for trainable network parameters.

 *Fields:*

 - `Parameter.data`: weights

 - `Parameter.grad`: gradient

"""
Parameter(x) = Parameter(x, nothing)

"""
    OUT_INIT_SCALE

Factor applied to the initialization of a conditioner block's output weight
([`ResidualBlock`](@ref), [`MLPBlock`](@ref)).

A coupling layer reads its conditioner's output as a (log-scale, shift) pair, so the output has
to be linear -- an activation there would confine both halves to one sign. Linear and
`glorot_uniform`, though, means the log-scale is spread over several units at initialization,
and a scale of `sigmoid(-3)` costs a factor of 20 in the inverse direction. That compounds
through depth, and a recursive coupling such as [`CouplingLayerHINT`](@ref) applied in reverse
can overflow `Float32` before it has been trained at all.

Shrinking the output weight puts every coupling layer near its identity map at initialization
and leaves training free to move away from it -- Glow zero-initializes the same weight, and
FrEIA scales the same quantity by the same order. It is a keyword on both constructors, so a
caller who wants the unshrunk initialization can ask for `out_scale=1`.
"""
const OUT_INIT_SCALE = 0.1f0



# Size and length for parameter types
size(x::Parameter) = size(x.data)
length(x::Parameter) = length(x.data)


Flux.@layer Parameter trainable=(data,)

"""
    clear_grad!(NL::NeuralNetLayer)

or

    clear_grad!(P::AbstractArray{Parameter, 1})

 Set gradients of each `Parameter` in the network layer to `nothing`.
"""
function clear_grad!(P::AbstractVector{<:Parameter})
    for j=1:length(P)
        P[j].grad = nothing
    end
end

function get_grads(p::Parameter)
    return Parameter(p.grad)
end

function get_grads(pvec::AbstractVector{<:Parameter})
    g = Array{Parameter, 1}(undef, length(pvec))
    for i = 1:length(pvec)
        g[i] = get_grads(pvec[i])
    end
    return g
end

get_params(p::Parameter) = p

function set_params!(pold::Parameter, pnew::Parameter)
    pold.data = pnew.data
    pold.grad = pnew.grad
end

function set_params!(pold::AbstractVector{<:Parameter}, pnew::AbstractVector{<:Parameter})
    for i = 1:length(pold)
        set_params!(pold[i], pnew[i])
    end
end

function set_params!(pold::AbstractVector{<:Parameter}, pnew::AbstractVector)
    for i = 1:length(pold)
        set_params!(pold[i], pnew[i])
    end
end



## Algebraic utilities for parameters

function dot(p1::Parameter, p2::Parameter)
    return dot(p1.data, p2.data)
end

function norm(p::Parameter)
    return norm(p.data)
end

function +(p1::Parameter, p2::Parameter)
    return Parameter(p1.data+p2.data)
end

function +(p1::Parameter, p2::T) where {T<:Real}
    return Parameter(p1.data+p2)
end

function +(p1::T, p2::Parameter) where {T<:Real}
    return p2+p1
end

function -(p1::Parameter, p2::Parameter)
    return Parameter(p1.data-p2.data)
end

function -(p1::Parameter, p2::T) where {T<:Real}
    return Parameter(p1.data-p2)
end

function -(p1::T, p2::Parameter) where {T<:Real}
    return -(p2-p1)
end

function -(p::Parameter)
    return Parameter(-p.data)
end

function *(p1::Parameter, p2::T) where {T<:Real}
    return Parameter(p1.data*p2)
end

function *(p1::T, p2::Parameter) where {T<:Real}
    return p2*p1
end

function /(p1::Parameter, p2::T) where {T<:Real}
    return Parameter(p1.data/p2)
end

function /(p1::T, p2::Parameter) where {T<:Real}
    return Parameter(p1/p2.data)
end

# Shape manipulation

par2vec(x::Parameter) = vec(x.data), size(x.data)


function vec2par(x::AbstractArray{T, 1}, s::NTuple{N, Int64}) where {T, N}
    return Parameter(reshape(x, s))
end

function par2vec(x::AbstractVector{<:Parameter})
    v = cat([vec(x[i].data) for i=1:length(x)]..., dims=1)
    s = cat([size(x[i].data) for i=1:length(x)]..., dims=1)
    return v, s
end

function vec2par(x::AbstractArray{T, 1}, s::Array{Any, 1}) where T
    xpar = Array{Parameter, 1}(undef, length(s))
    idx_i = 0
    for i = 1:length(s)
        xpar[i] = vec2par(x[idx_i+1:idx_i+prod(s[i])], s[i])
        idx_i += prod(s[i])
    end
    return xpar
end
