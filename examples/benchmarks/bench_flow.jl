# Throughput harness for coupling flows.
#
# Two things are measured, for two different reasons.
#
# `conv`   -- a convolutional flow, the architecture this package has always had. Its
#             parameterization is unchanged, so it is a like-for-like A/B against any earlier
#             commit and measures what the per-layer bookkeeping costs.
#
# `vector` -- a flow on plain `(dim, batch)` data, the shape a policy flow sees. The signature
#             to watch for here is the us/sample column: a flow whose cost is dominated by
#             per-layer overhead (kernel launches, host synchronizations, allocation) rather
#             than by arithmetic does not get cheaper per sample as the batch grows, so a flat
#             column means overhead-bound and a falling one means the work is real.
#
# Usage: julia --project=. examples/benchmarks/bench_flow.jl [quick] [conv|vector]

push!(LOAD_PATH, "@v#.#")   # BenchmarkTools lives in the default environment
using InvertibleNetworks, BenchmarkTools, Printf, Flux, Random

const QUICK = "quick" in ARGS
const WHICH = "conv" in ARGS ? (:conv,) : "vector" in ARGS ? (:vector,) : (:conv, :vector)

conv_flow(nc, depth, n_hidden) = InvertibleChain(
    Tuple(Iterators.flatten((ActNorm(nc; logdet=true),
                             CouplingLayerGlow(nc, n_hidden; logdet=true))
                            for _ = 1:depth)))

vector_flow(dim, depth, n_hidden) = InvertibleChain(
    Tuple(Iterators.flatten((ActNorm(dim; logdet=true),
                             CouplingLayerGlow(dim, n_hidden; logdet=true, dense=true))
                            for _ = 1:depth)))

# `minimum` rather than the mean: we are after the cost of the work, not of the GC that the
# allocations provoke, and the allocation counts are reported separately anyway.
function timed(f, args...)
    b = @benchmark $f($(args)...) samples=(QUICK ? 20 : 100) evals=1 seconds=(QUICK ? 1 : 3)
    return minimum(b.times)/1e3, b.memory/1024, b.allocs
end

function run_case(name, build, shape, depth, n_hidden, batches)
    @printf("\n%s  (depth=%d, n_hidden=%d)\n", name, depth, n_hidden)
    @printf("%-10s %-8s %-11s %-12s %-11s %-9s %-11s %-9s\n",
            "size", "batch", "fwd us", "fwd us/smp", "fwd KiB", "fwd alc", "grad us", "grad alc")
    println("-"^84)
    for batch in batches
        Random.seed!(1)
        flow = build()
        X = randn(Float32, shape..., batch)
        flow(X)                                    # initialize ActNorm, compile once

        fus, fkib, falc = timed(x -> flow(x), X)

        loss(m, x) = -log_likelihood(x, m)
        Flux.withgradient(m -> loss(m, X), flow)
        gus, _, galc = timed((f, x) -> Flux.withgradient(m -> loss(m, x), f), flow, X)

        @printf("%-10s %-8d %-11.1f %-12.4f %-11.1f %-9d %-11.1f %-9d\n",
                join(shape, "x"), batch, fus, fus/batch, fkib, falc, gus, galc)
    end
end

depth, n_hidden = 8, 32

if :conv in WHICH
    println("="^84)
    println("CONVOLUTIONAL FLOW  (unchanged architecture: a like-for-like A/B)")
    println("="^84)
    for (nx, nc) in (QUICK ? ((16, 4),) : ((16, 4), (32, 8)))
        run_case("conv $(nx)x$(nx)x$(nc)", () -> conv_flow(nc, depth, n_hidden),
                 (nx, nx, nc), depth, n_hidden, QUICK ? (1, 16) : (1, 8, 32))
    end
end

if :vector in WHICH
    println()
    println("="^84)
    println("VECTOR FLOW  (dim, batch) -- watch the us/sample column for flatness")
    println("="^84)
    for dim in (QUICK ? (8,) : (2, 8, 32, 128))
        run_case("vector dim=$dim", () -> vector_flow(dim, depth, n_hidden),
                 (dim,), depth, n_hidden, QUICK ? (1, 256, 4096) : (1, 64, 1024, 16384))
    end
end
