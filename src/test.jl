using Test
using Statistics
using LinearAlgebra

include("synthetic.jl")

@testset "gethair happy path" begin
    rng = MersenneTwister(3)

    z = genhair(4, npoints = 100, rng = rng)
    @test size(z) == (100, 2)
    # Distance between two adjacent points is close to 1
    dz = diff(z, dims = 1)
    print(size(dz))
    dist = mapslices(norm, dz, dims = 2)
    println([maximum(dist) minimum(dist)])

    @test all(dist .< 1.2)
    @test all(dist .> 0.8)

    cossin = dz ./ dist
    cosθ = dz[:, 1] ./ dist
    println(maximum(abs.(cossin)))

    dθ = diff(acos.(cosθ[:, 1]), dims = 1)
    @test abs(mean(dθ)) < 0.1
    @test π / 64 < std(dθ) < (π / 32 + 0.1)
end
