using LinearAlgebra
"""
    cg(A::Matrix{T}, b::Vector{T}, x₀::Vector{T}, e::Float64=1e-5) where {T <: AbstractFloat} -> AbstractVector

共轭梯度法求解 ``Ax=b``

# Aguments
- `A` 系数矩阵
- `b` 常数项
- `x₀` 迭代初值
- `e` 允许最大误差
# Examples
```julia
julia> A = Matrix{Float64}([2 2;2 5]);
julia> b = [6.0;3.0];
julia> x₀ = zeros(Float64, 2);
julia> x = cg(A,b,x₀)
2-element Array{Float64,1}:
  3.9999999999999987
 -0.9999999999999998
```
"""
function cg(A::Matrix{T}, b::Vector{T}, x₀::Vector{T}, e::Float64=1e-5) where T <: AbstractFloat
    @assert isposdef(A)  # 判断 A 是否为对称正定矩阵
    n = size(A, 1)
    @assert n == length(b) == length(x₀)  # 维度匹配
    d₀ = b - A*x₀
    r₀ = b - A*x₀
    for _ in 1:n
        if norm(r₀) < e
            break
        end
        α = (r₀' ⋅ r₀) / (d₀' * A * d₀)
        x = x₀ + α * d₀
        r = r₀ - α * A * d₀
        β = (r' ⋅ r) / (r₀' ⋅ r₀)
        d = r + β * d₀
        r₀ = r
        x₀ = x
        d₀ = d
    end
    x₀
end
