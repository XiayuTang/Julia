using Juno
using LinearAlgebra
"""
    cgm(A::AbstractMatrix{T}, b::AbstractVector{T}, x₀::AbstractVector{T}) where T <: AbstractFloat -> AbstractVector

Conjugate Gradient Method
Solving ``Ax=b``

# Aguments
- `A` 系数矩阵
- `b` 常数项
- `x₀` 迭代初值
"""
function cgm(A::Matrix{T}, b::Vector{T}, x₀::Vector{T}) where T <: AbstractFloat
    @assert isposdef(A)  # 判断 A 是否为对称正定矩阵
    n = size(A, 1)
    @assert n == length(b) == length(x₀)  # 维度匹配
    d₀ = b - A*x₀
    r₀ = b - A*x₀
    num = 0
    # println(r₀)
    # println(d₀)
    for _ in 1:n
        if r₀ == zeros(eltype(r₀), n)
            break
        end
        α = (r₀' ⋅ r₀) / (d₀' * A * d₀)
        # println(α)
        x = x₀ + α * d₀
        # println(x)
        r = r₀ + α * A * d₀
        # println(r)
        β = (r' ⋅ r) / (r₀' ⋅ r₀)
        # println(β)
        d = r + β * d₀
        # println(d)
        num += 1
        r₀ = r
        x₀ = x
        d₀ = d
    end
    # x₀
    nothing
end

function test()
    A = Matrix{Float64}[2 2;2 5]
    b = [6.0;3.0]
    x₀ = zeros(Float64, 2)
    cgm(A,b,x₀)
end
test()
