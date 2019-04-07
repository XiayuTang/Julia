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
