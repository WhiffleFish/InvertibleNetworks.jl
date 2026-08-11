export NeuralNetLayer, InvertibleNetwork, ReverseLayer, ReverseNetwork
export get_grads

abstract type Invertible end

# Base Layer and network types with property getters
abstract type NeuralNetLayer <: Invertible end
abstract type InvertibleNetwork <: Invertible end

# Concrete reversed types
struct Reversed <: Invertible
    I::Invertible
end

# Simple display
Base.show(io::IO, m::Invertible) = print(io, typeof(m))
Base.display(m::Invertible) = println(typeof(m))

# Propagation modes
_INet_modes = [:forward, :inverse, :backward, :backward_inv, :inverse_Y, :forward_Y,
               :jacobian, :jacobianInverse, :adjointJacobian, :adjointJacobianInverse]

_RNet_modes = Dict(:forward=>:inverse, :inverse=>:forward, :backward=>:backward_inv,
                   :inverse_Y=>:forward_Y, :forward_Y=>:inverse_Y)

# Actual call to propagation function. Dispatch through `Val` so the selected
# operation remains visible to inference; runtime `eval(sym)` made every
# `layer.forward(...)`/`network.backward(...)` call return `Any`.
function _predefined_mode(obj, mode::Val, args...; kwargs...)
    _apply_mode(mode, obj, args...; kwargs...)
end

for mode in _INet_modes
    @eval _apply_mode(::Val{$(Meta.quot(mode))}, obj, args...; kwargs...) =
        $mode(args..., obj; kwargs...)
end

# Base getproperty
getproperty(I::Invertible, s::Symbol) = _get_property(I, Val{s}())

_get_property(I::Invertible, ::Val{s}) where {s} = getfield(I, s)
_get_property(R::Reversed, ::Val{:I}) = getfield(R, :I)
_get_property(R::Reversed, ::Val{s}) where s = _get_property(R.I, Val{s}())

for m ∈ _INet_modes
    @eval _get_property(I::Union{InvertibleNetwork,NeuralNetLayer}, ::Val{$(Meta.quot(m))}) =
        (args...; kwargs...) -> _predefined_mode(I, Val($(Meta.quot(m))), args...; kwargs...)
end

for (m, k) ∈ _RNet_modes
    @eval _get_property(R::Reversed, ::Val{$(Meta.quot(m))}) = _get_property(R.I, Val{$(Meta.quot(k))}())
end

# Type conversions
function convert_params!(::Type{T}, obj::Invertible) where T
    _convert_fields!(T, obj)
end

@generated function _convert_fields!(::Type{T}, obj::I) where {T,I<:Invertible}
    conversions = [:(convert_params!(T, getfield(obj, $i))) for i in 1:fieldcount(I)]
    return Expr(:block, conversions..., :(nothing))
end

convert_params!(::Type{T}, p::Parameter) where {T} = convert_param!(T, p)
function convert_params!(::Type{T}, values::AbstractArray) where {T}
    for value in values
        convert_params!(T, value)
    end
    return nothing
end
function convert_params!(::Type{T}, values::Tuple) where {T}
    for value in values
        convert_params!(T, value)
    end
    return nothing
end
convert_params!(::Type, ::Any) = nothing

input_type(x::AbstractArray) = eltype(x)
input_type(x::Tuple) = eltype(x[1])

# Reverse
# For networks and layers not needing the tag
tag_as_reversed!(I::Invertible, ::Bool) = I

reverse(L::NeuralNetLayer) = Reversed(tag_as_reversed!(deepcopy(L), true))
reverse(N::InvertibleNetwork) = Reversed(tag_as_reversed!(deepcopy(N), true))
reverse(RL::Reversed) = tag_as_reversed!(deepcopy(RL.I), false)

"""
    P = get_params(NL::Invertible)

 Returns a cell array of all parameters in the network or layer. Each cell
 entry contains a reference to the original parameter; i.e. modifying
 the paramters in `P`, modifies the parameters in `NL`.
"""
function get_params(I::Invertible)
    params = Vector{Parameter}(undef, 0)
    for (f, tp) ∈ zip(fieldnames(typeof(I)), typeof(I).types)
        p = getfield(I, f)
        if tp <: Parameter
            append!(params, [p])
        else
            append!(params, get_params(p))
        end
    end
    params
end

get_params(x) = Array{Parameter}(undef, 0)
get_params(A::Array{T}) where T<:Union{Invertible, Nothing} = vcat([get_params(A[i]) for i in 1:length(A)]...)
get_params(values::Tuple) = vcat((get_params(value) for value in values)...)
get_params(A::Matrix{T}) where T<:Union{Invertible, Nothing} = vcat([get_params(A[i, j]) for i=1:size(A, 1) for j in 1:size(A, 2)]...)
get_params(RN::Reversed) = get_params(RN.I)

# reset! parameters
"""
    P = reset!(NL::Invertible)

 Resets stateful layers and clears cached parameter gradients without changing
 the concrete parameter storage.
"""
function reset!(I::Invertible)
    _reset_fields!(I)
    return I
end

@generated function _reset_fields!(I::T) where {T<:Invertible}
    resets = [:(reset_value!(getfield(I, $i))) for i in 1:fieldcount(T)]
    return Expr(:block, resets..., :(nothing))
end

reset_value!(p::Parameter) = (p.grad = nothing; nothing)
reset_value!(I::Invertible) = (reset!(I); nothing)
function reset_value!(values::Union{AbstractArray,Tuple})
    for value in values
        reset_value!(value)
    end
    return nothing
end
reset_value!(::Any) = nothing

function reset!(AI::AbstractArray{<:Invertible})
    for I in AI
        reset!(I)
    end
    return AI
end

# Clear grad functionality for reversed layers/networks
"""
    P = clear_grad!(NL::Invertible)

 Resets the gradient of all the parameters in NL
"""
clear_grad!(I::Invertible) = clear_grad!(get_params(I))

# Get gradients
"""
    P = get_grads(NL::Invertible)

 Returns a cell array of all parameters gradients in the network or layer. Each cell
 entry contains a reference to the original parameter's gradient; i.e. modifying
 the paramters in `P`, modifies the parameters in `NL`.
"""
get_grads(I::Invertible) = [Parameter(p.grad) for p ∈ get_params(I)]
get_grads(A::Array{Union{Invertible, Nothing}}) = vcat([get_grads(A[i]) for i in 1:length(A)]...)
get_grads(RL::Reversed)= get_grads(RL.I)
get_grads(::Nothing) = []

# Set parameters
function set_params!(N::Invertible, θnew::AbstractVector{<:Parameter})
    set_params!(get_params(N), θnew)
end

# Set parameters with BSON loaded params
function set_params!(N::Invertible, θnew::Array{Any, 1})
    set_params!(get_params(N), θnew)
end

# Return the trainable arrays in the same stable order as `get_params`.
# The ChainRules rule for this function maps their cotangents back onto the
# nested model structure expected by Flux's explicit optimiser interface.
parameter_data(net::Invertible) = getfield.(get_params(net), :data)

# Make invertible nets callable objects
(net::Invertible)(X::AbstractArray{T,N} where {T, N}) = forward_net(net, X, parameter_data(net))
forward_net(net::Invertible, X::AbstractArray{T,N}, ::Any) where {T, N} = net.forward(X)

# Make conditional invertible nets callable objects
(net::Invertible)(X::AbstractArray{T,N}, Y::AbstractArray{T,N}) where {T, N} = forward_net(net, X, Y, parameter_data(net))
forward_net(net::Invertible, X::AbstractArray{T,N}, Y::AbstractArray{T,N}, ::Any) where {T, N} = net.forward(X,Y)
