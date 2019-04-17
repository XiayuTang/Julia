A = Matrix{Float64}([2 2;2 5])
b = [6.0;3.0]
x₀ = zeros(Float64, 2)
x = cgm(A,b,x₀)


