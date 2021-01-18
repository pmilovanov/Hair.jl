using Test
using Statistics, LinearAlgebra, Random
using Hair
H=Hair


@testset "gethair happy path" begin
  rng = MersenneTwister(3)

  z = H.genhair(4, npoints = 100, rng = rng)
  @test size(z) == (100, 2)
  # Distance between two adjacent points is close to 1
  dz = diff(z, dims = 1)
  dist = mapslices(norm, dz, dims = 2)

  @test all(dist .< 1.2)
  @test all(dist .> 0.8)

  cossin = dz ./ dist
  cosθ = dz[:, 1] ./ dist

  dθ = diff(acos.(cosθ[:, 1]), dims = 1)
  @test abs(mean(dθ)) < 0.1
  @test π / 64 < std(dθ) < (π / 32 + 0.1)
end
