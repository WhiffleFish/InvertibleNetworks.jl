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
            @test inferred_type(InvertibleNetworks._forward, typeof(X), Nothing,
                                typeof(spline_layer), Val{:sample}) ==
                  Tuple{Array{Float32,4},Vector{Float32}}
            # ...and the conditional pass has to infer just as concretely
            ctx_layer = CouplingLayerSpline(4, 8; spline=kind, logdet=true, n_ctx=2)
            @test inferred_type(InvertibleNetworks._forward, typeof(X), Array{Float32,4},
                                typeof(ctx_layer), Val{:sample}) ==
                  Tuple{Array{Float32,4},Vector{Float32}}
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
        for layer in (MLPBlock(4, 8), MLPBlock(4, 8; fan=true))
            T = typeof(layer)
            @test isempty([f for (f, ft) in zip(fieldnames(T), T.types) if !isconcretetype(ft)])
        end
    end
end

@testset "Dense/vector path inference" begin
    V = randn(Float32, 8, 32)      # (dim, batch)

    # The hot path is `block_forward`/`block_forward_save`, which pass a `Val` literal. The
    # public `forward(X, B; save)` takes `save` as a runtime `Bool`, so its return type is a
    # union over both -- true of `ResidualBlock` too, and why the `block_*` entry points exist.
    mlp = MLPBlock(8, 16; fan=true)
    @test inferred_type(InvertibleNetworks.block_forward, typeof(V), typeof(mlp)) ==
          Matrix{Float32}
    @test inferred_type(InvertibleNetworks._forward, typeof(V), typeof(mlp), Val{false}) ==
          Matrix{Float32}
    @test isconcretetype(inferred_type(InvertibleNetworks._forward, typeof(V), typeof(mlp),
                                       Val{true}))

    # A dense coupling layer on plain vectors, with and without the log-determinant
    @test inferred_type(InvertibleNetworks.forward, typeof(V),
                        typeof(CouplingLayerGlow(8, 16; dense=true))) == Matrix{Float32}
    @test inferred_type(InvertibleNetworks.forward, typeof(V),
                        typeof(CouplingLayerGlow(8, 16; dense=true, logdet=true))) ==
          Tuple{Matrix{Float32},Float32}
    @test inferred_type(InvertibleNetworks._forward, typeof(V),
                        typeof(CouplingLayerGlow(8, 16; dense=true, logdet=true)),
                        Val{:sample}) == Tuple{Matrix{Float32},Vector{Float32}}

    # The chain now accumulates per sample internally and reduces once, which must not widen
    # the scalar it returns.
    flow = InvertibleChain(ActNorm(8; logdet=true),
                           CouplingLayerGlow(8, 16; logdet=true, dense=true))
    @test inferred_type(InvertibleNetworks.forward, typeof(V), typeof(flow)) ==
          Tuple{Matrix{Float32},Float32}
    @test inferred_type(InvertibleNetworks._chain_forward, typeof(V), Nothing, typeof(flow),
                        Val{true}) == Tuple{Matrix{Float32},Float32}
end

@testset "Channel views" begin
    for X in (randn(Float32, 8, 8, 4, 3), randn(Float32, 8, 6))
        a, b = InvertibleNetworks.channel_halves(X)
        ta, tb = InvertibleNetworks.tensor_split(X)
        @test a == ta && b == tb
        @test size(a) == size(ta) && size(b) == size(tb)
        # Views, not copies: writing through one is visible in the parent
        a[1] = 42f0
        @test X[1] == 42f0
    end
end
