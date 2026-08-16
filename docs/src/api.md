
# Invertible Networks API reference

```@autodocs
Modules = [InvertibleNetworks]
Order  = [:function]
Pages = ["neuralnet.jl", "parameter.jl"]
```

## Activation functions

```@autodocs
Modules = [InvertibleNetworks]
Order   = [:function]
Pages = ["activation_functions.jl"]
```

## Dimensions manipulation

```@autodocs
Modules = [InvertibleNetworks]
Order   = [:function]
Pages = ["dimensionality_operations.jl"]
```

## Layers

```@autodocs
Modules = [InvertibleNetworks]
Order  = [:type]
Filter = t -> t<:NeuralNetLayer
```

## Spline transforms

```@autodocs
Modules = [InvertibleNetworks]
Order  = [:type, :function]
Pages = ["splines.jl"]
```

## Networks

```@autodocs
Modules = [InvertibleNetworks]
Order   = [:type]
Filter = t -> t<:InvertibleNetwork
```

## Augmented flows

```@autodocs
Modules = [InvertibleNetworks]
Order   = [:function]
Pages = ["augmented_flow.jl"]
```

## Latent distributions

The base distribution a flow pushes forward is pluggable through the `base` keyword of every
likelihood entry point. A base with bounded support additionally carries a domain invariant,
checked against the network before any density is computed.

```@autodocs
Modules = [InvertibleNetworks]
Order   = [:type, :function]
Pages = ["latent_distributions.jl"]
```

## AD Integration

```@autodocs
Modules = [InvertibleNetworks]
Order  = [:function]
Pages = ["chainrules.jl"]
```