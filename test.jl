# push!(LOAD_PATH,".")
#
# using LinearEquationsSolve
function fib(n)
    a, b = 1, 1
    for _ in 1:n-1
        a, b = b, a+b
    end
    b
end
