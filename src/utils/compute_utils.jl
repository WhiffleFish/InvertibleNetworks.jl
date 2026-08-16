using CUDA

export convert_cu, cuzeros, cuones, array_of_array, chain_lr
export householder_matrix, householder_reflector, apply_channel_matrix

convert_cu(in_a, X) =  X isa CuArray ? cu(in_a) : in_a
cuzeros(::Array{T, N}, a::Vararg{Int, N2}) where {T, N, N2} = zeros(T, a...)
cuzeros(::CuArray{T, N}, a::Vararg{Int, N2}) where {T, N, N2} = CUDA.zeros(T, a...)
cuzeros(x, a::Tuple) = cuzeros(x, a...)
cuones(::Array{T, N}, a::Vararg{Int, N2}) where {T, N, N2} = ones(T, a...)
cuones(::CuArray{T, N}, a::Vararg{Int, N2}) where {T, N, N2} = CUDA.ones(T, a...)
cuones(x, a::Tuple) = cuones(x, a...)

array_of_array(::Array, args...) = Array{Array}(undef, args...)
array_of_array(::CuArray, args...) = Array{CuArray}(undef, args...)

# Statically shaped indexing/reduction tuples used throughout convolutional
# layers. The former comprehensions produced `Vector{Any}` because they mixed
# integers and `Colon()`, which poisoned inference at every reshape/index.
@inline channel_indices(::Val{N}) where {N} = ntuple(i -> i == N - 1 ? Colon() : 1, Val(N))
@inline batch_reduction_dims(::Val{N}) where {N} = ntuple(i -> i == N - 1 ? N : i, Val(N - 1))

# for 1x1 Conv
gemm_outer!(out::Matrix{T}, tmp::Vector{T}, v::Vector{T}) where T = LinearAlgebra.BLAS.gemm!('N', 'T', T(1), tmp, v, T(1), out)
gemm_outer!(out::CuMatrix{T}, tmp::CuVector{T}, v::CuVector{T}) where T = CUDA.cuBLAS.gemm!('N', 'T', T(1), tmp, v, T(1), out)

function chain_lr(x::AbstractMatrix{T}, vi::Vararg{AbstractVector{T}, N}) where {T, N}
    out = T(1) .* x
    tmp = cuzeros(vi[1], size(x, 1))
    for v=vi
        n = -2/dot(v, v)
        mul!(tmp, out, v)
        rmul!(tmp, n)
        gemm_outer!(out, tmp, v)
    end
    out
end

# `k x k` identity in whatever array type `v` lives in. The diagonal is written through a
# strided view rather than element by element, so it needs no scalar indexing on the GPU.
function eye_like(v::AbstractVector{T}, k::Integer) where T
    E = cuzeros(v, k, k)
    E[1:k+1:end] .= one(T)
    return E
end

# The single Householder reflector `I - 2vv'/(v'v)`. Only ever `k x k`, and only used to
# assemble the derivative in `jacobian`, so it is written for clarity rather than for speed.
function householder_reflector(v::AbstractVector{T}) where T
    H = eye_like(v, length(v))
    H .-= (T(2)/dot(v, v)) .* (v * adjoint(v))
    return H
end

"""
    Q = householder_matrix(v1, v2, ...)

 The `k x k` orthogonal matrix that [`chain_lr`](@ref) applies on the right, i.e.
 `chain_lr(x, v...) == x * householder_matrix(v...)`.

 The product of the reflectors is the *same matrix for every sample in the batch*, so forming
 it once and contracting the whole batch against it turns a per-sample chain of rank-1 updates
 into one GEMM. Building it costs `O(k^2)` regardless of the batch size.
"""
householder_matrix(vi::Vararg{AbstractVector{T}, N}) where {T, N} =
    chain_lr(eye_like(vi[1], length(vi[1])), vi...)

"""
    Y = apply_channel_matrix(X, Q)

 Mix the channel dimension of an ND tensor by a `k x k` matrix:
 `Y[s, c', b] = sum_c X[s, c, b] * Q[c, c']`, with `s` ranging over the spatial dimensions.

 The two cases below are the same contraction, but they need different BLAS shapes to run at
 speed, which is the whole point of splitting them:

 - No spatial extent (`(k, b)` vectors, or the `(1, 1, k, b)` tensors a flow on tabular or
   vector data uses): channels and batch are the only axes, so the entire batch is a single
   `k x k` by `k x b` GEMM.

 - With spatial extent: batching is `NNlib.batched_mul`, one `(s x k) * (k x k)` GEMM per
   sample. Those are large enough to keep BLAS busy, and unlike flattening spatial and batch
   together it needs no `permutedims` of the data -- which, measured, costs more than it saves
   because the transpose moves as much memory as the multiply reads.
"""
function apply_channel_matrix(X::AbstractArray{T, N}, Q::AbstractMatrix{T}) where {T, N}
    k, batchsize = size(X, N-1), size(X, N)
    spatial = prod(ntuple(i -> size(X, i), Val(N-2)))
    if spatial == 1
        return reshape(adjoint(Q) * reshape(X, k, batchsize), size(X))
    end
    Xs = reshape(X, spatial, k, batchsize)
    return reshape(NNlib.batched_mul(Xs, reshape(Q, k, k, 1)), size(X))
end
