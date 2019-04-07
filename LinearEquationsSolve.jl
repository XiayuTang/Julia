module LinearEquationsSolve
export diagchange!,jacobi
using LinearAlgebra


"""
    isrowzero(A)

判断矩阵A的每一行是否全为零
"""
function isrowzero(A::AbstractMatrix)
    n = size(A,1)
    o = zeros(eltype(A),n)
    for i ∈ 1:n
        if A[i,:] == o
            return true
        end
    end
    false
end


"""
    iscolzero(A)

判断矩阵A的每一列是否全为零
"""
function iscolzero(A::AbstractMatrix)
    n = size(A,2)
    o = zeros(eltype(A),n)
    for i ∈ 1:n
        if A[:,i] == o
            return true
        end
    end
    false
end


"""
    diagchange!(A)

若矩阵A的对角元的某几个为0，
则通过初等行变换将0转换为非零元素
"""
function diagchange!(A::AbstractMatrix{T}) where T <: AbstractFloat
    m,n = size(A)
    @assert m == n  # 要求 A为方阵
    @assert !isrowzero(A)
    @assert !iscolzero(A)
    for i ∈ 1:n
        if isapprox(A[i,i],0,atol=1e-6)
            # 若第 i个对角元素接近于 0，则寻找该对角元所处的那一列
            # 中的绝对值最大元所处的行数
            k = argmax(abs.(A[:,i]))
            if abs(A[k,i]) < 1e-6
                error("矩阵A是病态的！")
            end
            A[[i,k],:] = A[[k,i],:]  # 交换两行
        end
    end
    nothing
end


"""
    jacobi(A,b,x⁰,ϵ,maxiter) -> (x,n)

Jacobi迭代法求解线性方程组 ``Ax = b``

# Aguments
- `A::AbstractMatrix{AbstractFloat}`  系数矩阵
- `b::AbstractVector{AbstractFloat}`  常数项
- `x⁰::AbstractVector{AbstractFloat}`  迭代初值
- `ϵ::AbstractFloat=1e-5`  允许误差
- `maxiter::Int=100`  最大迭代次数
- `x::AbstractVector{AbstractFloat}`  方程组的近似解
- `n`  迭代次数

# Mathematical Theory
将系数矩阵``A``分解为
```math
A = L + D + U
```
其中``L``是下三角矩阵，由``A``的对角线之下的元素组成，``D``是对角矩阵，由``A``的对角线元素组成，``U``是上三角矩阵，由``A``的对角线之上的元素组成。

给定迭代初值`x⁰`，则第``k+1``次迭代可表示为
```math
Dx^{k+1} + (L+U)x^{k} = b
```
即
```math
x^{k+1} = -D^{-1}(L+U)x^k + D^{-1}b
```
令
```math
 r^k = b - Ax^k
```
则
```math
b = r^k + Ax^k = r^k + (L+D+U)x^k
```
于是Jacobi迭代形式可写为
```math
x^{k+1} = x^k + D^{-1}r^k
```
# Example
```jldoctest
>julia a = [2.0 1.0; 1.0 1.0];
>julia b = [3.0; 2.0];
>julia x⁰ = [0.7;0.7];
>julia x,n = jacobi(a,b,x⁰);
>julia x
2-element Array{Float64,1}:
 0.9999977111816405
 0.9999977111816405
>julia n
34
```
"""
function jacobi(A::AbstractMatrix{T},b::AbstractVector{T},x⁰::AbstractVector{T},ϵ::AbstractFloat=1e-5,maxiter::Int=100) where T <: AbstractFloat
    diagchange!(A)
    n = 0
    D = Diagonal(diag(A))
    Dinv = inv(D)
    while true
        r⁰ = b - A*x⁰
        x¹ = x⁰ + Dinv*r⁰
        n += 1
        if norm(x¹-x⁰) < ϵ
            return x¹,n
        end
        if n > maxiter
            println("超过最大迭代次数！")
            return nothing
        end
        x⁰ = x¹
    end
end
end  # module
