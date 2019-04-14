# PSO算法求解方程组的根

mutable struct Particle
    n::Integer  # the dimensionality of the search space
    x::Vector{Float64}  # the current position
    v::Vector{Float64}  # the velocity
    p::Vector{Float64}  # the previous best position
    pbest::Float64  # the value of the best function result
end
Particle(n, vmax) = Particle(n, rand(n), 2*vmax .* rand(n) .- vmax,)

mutable struct Swarm
    n::Int64  # the dimensionality of the search space
    func::Function  # the function
    particles::Vector{Particle}  # particles
    vmax::Float64  # the limits of velocity v ∈ [-vmax, vmax]
    𝝓₁::Float64  # a random number
    𝝓₂::Float64  # a random number
    pg::Vector{Float64}  # the particle in the neighborhood with the best success
end

Swarm(n::Int64, func::Function, particles::Vector{Particle}, vmax::Float64=5.0, 𝝓₁::Float64=2.1, 𝝓₂::Float64=2.1) = Swarm(n, func, particles, vmax, 𝝓₁, 𝝓₂, zeros(Float64, n))
