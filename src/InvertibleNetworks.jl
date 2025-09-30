# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020
# Copyright: Georgia Institute of Technology, 2020

"""
    InvertibleNetworks

Building blocks for invertible neural networks in Julia.

This package provides memory-efficient building blocks for invertible neural networks
with hand-derived gradients, Jacobians, and log-determinants. It includes support
for Flux integration, Zygote and ChainRules automatic differentiation, and GPU support.

## Key Features

- Memory efficient building blocks for invertible neural networks
- Hand-derived gradients, Jacobians J, and log|J|
- Flux integration with support for Zygote and ChainRules
- GPU support via CuArray
- Various examples of invertible neural networks, normalizing flows, 
  variational inference, and uncertainty quantification

## Main Components

- **Layers**: ActNorm, Conv1x1, CouplingLayerGlow, CouplingLayerHINT, etc.
- **Networks**: NetworkGlow, NetworkHINT, NetworkHyperbolic, etc.
- **Utilities**: Parameter management, objective functions, dimensionality operations

## Quick Start

```julia
using InvertibleNetworks, Flux

# Create a simple activation normalization layer
an = ActNorm(10; logdet=true)

# Forward pass
X = randn(Float32, 64, 64, 10, 4)
Y, logdet = an.forward(X)

# Inverse pass
X_reconstructed = an.inverse(Y)
```
"""
module InvertibleNetworks

# Core dependencies
using LinearAlgebra, Random
using Statistics, Wavelets
using JOLI
using NNlib, Flux, ChainRulesCore

# Overloads and reexports
import Base.size, Base.length, Base.getindex, Base.reverse, Base.reverse!, Base.getproperty
import Base.+, Base.*, Base.-, Base./
import LinearAlgebra.dot, LinearAlgebra.norm, LinearAlgebra.adjoint
import Flux.glorot_uniform
import CUDA: CuArray

export clear_grad!, glorot_uniform

# Getters for DenseConvDims fields
# (need to redefine here as they are not public methods in NNlib)
input_size(c::DenseConvDims) = c.I
kernel_size(::DenseConvDims{N,K,S,P,D}) where {N,K,S,P,D} = K
channels_in(dcd::DenseConvDims{N,K,S,P,D}) where {N,K,S,P,D} = dcd.channels_in
channels_out(dcd::DenseConvDims{N,K,S,P,D}) where {N,K,S,P,D} = dcd.channels_out

"""
    dense_conv_dims(X::AbstractArray{T, N}, W::AbstractArray{T, N}; 
                    stride=1, padding=1, nc=nothing) where {T, N}

Create DenseConvDims for convolution operations.

# Arguments
- `X`: Input tensor
- `W`: Weight tensor  
- `stride`: Stride for convolution (default: 1)
- `padding`: Padding for convolution (default: 1)
- `nc`: Number of channels (default: inferred from W)

# Returns
- `DenseConvDims` object for the convolution operation
"""
function dense_conv_dims(X::AbstractArray{T, N}, W::AbstractArray{T, N}; 
                        stride=1, padding=1, nc=nothing) where {T, N}
    sw = size(W)
    isnothing(nc) && (nc = sw[N-1])
    sx = (size(X)[1:N-2]..., nc, size(X)[end])
    return DenseConvDims(sx, sw; stride=Tuple(stride for i=1:N-2), 
                        padding=Tuple(padding for i=1:N-2))
end

# Legacy alias for backward compatibility
const DCDims = dense_conv_dims

# Utils
include("utils/parameter.jl")
include("utils/objective_functions.jl")
include("utils/dimensionality_operations.jl")
include("utils/activation_functions.jl")
include("utils/test_distributions.jl")
include("utils/neuralnet.jl")
include("utils/invertible_network_sequential.jl")
# AD rules
include("utils/chainrules.jl")

# Single network layers (invertible and non-invertible)
include("conditional_layers/conditional_layer_residual_block.jl")
include("layers/layer_flux_block.jl")
include("layers/layer_residual_block.jl")
include("layers/layer_resnet.jl")
include("layers/layer_affine.jl")
include("layers/invertible_layer_actnorm.jl")
include("layers/invertible_layer_conv1x1.jl")
include("layers/invertible_layer_basic.jl")
include("layers/invertible_layer_irim.jl")
include("layers/invertible_layer_glow.jl")
include("layers/invertible_layer_hyperbolic.jl")
include("layers/invertible_layer_hint.jl")

# Invertible network architectures
include("networks/invertible_network_hint_multiscale.jl")
include("networks/invertible_network_irim.jl")  # i-RIM: Putzky and Welling (2019)
include("networks/invertible_network_glow.jl")  # Glow: Dinh et al. (2017), Kingma and Dhariwal (2018)
include("networks/invertible_network_hyperbolic.jl")    # Hyperbolic: Lensink et al. (2019)

# Conditional layers and nets
include("conditional_layers/conditional_layer_glow.jl")
include("conditional_layers/conditional_layer_hint.jl")
include("networks/invertible_network_conditional_glow.jl")
include("networks/invertible_network_conditional_hint.jl")
include("networks/invertible_network_conditional_hint_multiscale.jl")

include("networks/summarized_net.jl")

# Jacobians
include("utils/jacobian.jl")

# GPU utilities
include("utils/compute_utils.jl")

end
