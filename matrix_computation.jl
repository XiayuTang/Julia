# MATRIX COMPUTATIONS 4th Edition Gene H. Golub, Charles F. Van Loan
# Chapter 1 Matrix Multiplication
"""
    dot(x, y) -> c

Algorithm 1.1.1 (Dot Product) If `x, y ∈ ℝⁿ`, then this algorithm computes their dot
product `c = xᵀy`.
"""
function dot(x::AbstractVector{T}, y::AbstractVector) where T <: Signed
    n = length(x)
    @assert n == length(y)
    c = zero(eltype(y))
    for i in 1:n
        c += x[i] * y[i]
    end
    c
end
function dot(x::AbstractVector{T}, y::AbstractVector) where T <: AbstractFloat
    n = length(x)
    @assert n == length(y)
    c = 0.0
    for i in 1:n
        c += x[i] * y[i]
    end
    c
end

"""
    saxpy!(x, y, a) -> y

Algorithm 1.1.2 (Saxpy) If `x`, `y` ∈ ℝⁿ and `a` ∈ ℝ, then this algorithm overwrites
`y` with ``y + ax``.
"""
function saxpy!(x::AbstractVector, y::AbstractVector, a::Real)
    n = length(x)
    @assert x == length(y)
    for i in 1:n
        y[i] += a * x[i]
    end
end

"""
    rogaxpy!(A, x, y) -> y

Algorithm 1.1.3 (Row-Oriented Gaxpy) If `A` ∈ ``ℝ^{m×n}``, `x` ∈ ℝⁿ, and `y` ∈ ℝᵐ, then this algorithm overwrites `y` with ``Ax + y``.
"""
function rogaxpy!(A::AbstractMatrix, x::AbstractVector, y::AbstractVector)
    n = length(x)
    m = length(y)
    @assert (m, n) == size(A)
    @inbounds for i in 1:n, j in 1:m
        y[i] += A[i, j] * x[j]
    end
end

"""
    cogaxpy!(A, x, y) -> y

Algorithm 1.1.4 (Column-Oriented Gaxpy) If `A` ∈ ``ℝ^{m×n}``, `x` ∈ ℝⁿ, and
`y` ∈ ℝᵐ, then this algorithm overwrites `y` with  ``Ax+y``.
"""
function cogaxpy!(A::AbstractMatrix, x::AbstractVector, y::AbstractVector)
    n = length(x)
    m = length(y)
    @assert (m, n) == size(A)
    @inbounds for j in 1:n, i in 1:m
        y[i] += A[i, j] * x[j]
    end
end

"""
    matmul!(A, B, C) -> C

Algorithm 1.1.5 (ijk Matrix Multiplication) If `A` ∈ ``ℝ^{m×r}``, `B` ∈ ``ℝ^{r×n}``, and `C` ∈ ``ℝ^{m×n}`` are give, and this algorithm overwrites `C`
with ``C + AB``.
"""
function matmul!(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix)
    m, r = size(A)
    r_, n = size(B)
    m_, n_ = size(C)
    @assert r == r_ && m == m_ && n == n_
    @inbounds for j in 1:n, k in 1:r, i in 1:m
        C[i, j] += A[i, k] * B[k, j]
    end
end
