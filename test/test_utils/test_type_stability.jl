using InvertibleNetworks, Test

# Return type of `f(args...)`, asserted to be a single inferred type.
inferred_type(f, argtypes...) = only(Base.return_types(f, Tuple{argtypes...}))

@testset "Hot-path inference" begin
    X = randn(Float32, 8, 8, 4, 2)

    squeezer = ShuffleLayer()
    @test inferred_type(squeezer.forward, typeof(X)) == Array{Float32,4}

    layer = CouplingLayerGlow(4, 8; logdet=false)
    parameter = first(get_params(layer))
    @test fieldtype(typeof(parameter), :data) == typeof(parameter.data)
    @test fieldtype(typeof(parameter), :grad) == Union{Nothing,typeof(parameter.data)}
    @test inferred_type(InvertibleNetworks.forward, typeof(X), typeof(layer)) == Array{Float32,4}

    network = NetworkGlow(4, 8, 2, 2; logdet=false)
    @test inferred_type(InvertibleNetworks.forward, typeof(X), typeof(network)) == Array{Float32,4}

    # `logdet=true` is the default for the flow networks, so it is the configuration that
    # matters most. The accumulated logdet must infer as T, not Union{T,Int}: seeding the
    # accumulator with a literal `0` silently widens the whole return type.
    @testset "logdet=true" begin
        logdet_layer = CouplingLayerGlow(4, 8; logdet=true)
        @test inferred_type(InvertibleNetworks.forward, typeof(X), typeof(logdet_layer)) ==
              Tuple{Array{Float32,4},Float32}

        logdet_norm = ActNorm(4; logdet=true)
        @test inferred_type(InvertibleNetworks._forward, typeof(X), typeof(logdet_norm), Val{true}) ==
              Tuple{Array{Float32,4},Float32}

        logdet_network = NetworkGlow(4, 8, 2, 2)
        @test logdet_network.logdet
        @test inferred_type(InvertibleNetworks.forward, typeof(X), typeof(logdet_network)) ==
              Tuple{Array{Float32,4},Float32}

        # The spline kind is a type parameter, so the branch between the families has to be
        # resolved at compile time rather than boxing every knot array.
        for kind in (:rqs, :circular, :lrs)
            spline_layer = CouplingLayerSpline(4, 8; spline=kind, logdet=true)
            @test inferred_type(InvertibleNetworks.forward, typeof(X), typeof(spline_layer)) ==
                  Tuple{Array{Float32,4},Float32}
            @test inferred_type(InvertibleNetworks._forward, typeof(X), typeof(spline_layer),
                                Val{:sample}) == Tuple{Array{Float32,4},Vector{Float32}}
            @test inferred_type(InvertibleNetworks.forward, typeof(X),
                                typeof(SplineLayer(4; spline=kind, logdet=true))) ==
                  Tuple{Array{Float32,4},Float32}
        end
    end

    # Layers whose parameters live in a type parameter must not leak an abstract field
    # type, or every `getfield` on them boxes.
    @testset "concrete fields" begin
        for layer in (ActNorm(4), Conv1x1(4), ResidualBlock(4, 8), CouplingLayerBasic(4, 8),
                      CouplingLayerGlow(4, 8), CouplingLayerIRIM(4, 8), AffineLayer(8, 8, 4),
                      CouplingLayerSpline(4, 8), CouplingLayerSpline(4, 8; spline=:circular),
                      SplineLayer(4))
            T = typeof(layer)
            abstract_fields = [f for (f, ft) in zip(fieldnames(T), T.types) if !isconcretetype(ft)]
            @test isempty(abstract_fields)
        end
    end
end
