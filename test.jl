# push!(LOAD_PATH,".")
using LinearEquationsSolve
using LinearAlgebra
# a = [2.0 1.0; 1.0 1.0];
# b = [3.0; 2.0];
# x⁰ = [0.7;0.7];
# x,n = gauss_seidel(a,b,x⁰)
# a = ones(4)
# for i in 1:4
#     a[i] = i
# end
# a
function generateA(n::Int)
    @assert n%2 == 0
    temp = -1*ones(n-1)
    A = diagm(0=>3*ones(12),1=>temp, -1=>temp);
    for i in 1:12
        if i==n/2 || i==(n/2+1)
            continue
        end
        A[i, n+1-i] = 1/2
    end
    A
end
function int(f)
    floor(Int,f)
end
function generateb(n::Int)
    @assert n%2==0
    b = 1.5*ones(n)
    b[1] = 2.5
    b[end] = 2.5
    b[int(n/2)] = 1.0
    b[int(n/2)+1] = 1.0
    b
end
n = 200
A = generateA(n)
b = generateb(n)
cg(A, b, zeros(n))
