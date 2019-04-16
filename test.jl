# push!(LOAD_PATH,".")
#
# using LinearEquationsSolve
# A = [1 2; 3 4; 5 6]
# @inbounds for i = 1:3, j = 1:2
# println(A[i, j])
# end
# function g(A)
#       m,n = size(A)
#       @simd for i=1:m
#             @inbounds println(A[i,:])
#       end
# end
# # a0 = ones(Int,19)
# # b0 = ones(Int,19)
# g(A)
# using LinearAlgebra
# @time begin
# for _ in 1:100000
#         s = 0.0
#         x = rand(1000)
#         y = rand(1000)
#         s = dot(x,y)
# end
# end
# using LinearAlgebra
# Diagonal
# function test1()
#     s = 0.0
#     for i in 1:100
#         s += 1/i
#     end
# end
# @time test1()
# using LinearAlgebra
# diagind
using LinearAlgebra
function test()
    for _ in 1:100
        A = rand(3000, 2000)
        B = rand(2000, 1000)
        A * B
    end
    nothing
end
@time test() 
