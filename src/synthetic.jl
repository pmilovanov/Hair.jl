using Interpolations
using Random


function genhair(
  nknots::Int = 4;
  npoints::Int = 100,
  σ_r::Real = 0.1,
  σ_θ::Real = π / 32,
  σ₀_θ::Real = π / 6,
  rng::AbstractRNG = Random.default_rng(),
)
  @assert nknots >= 0
  @assert npoints > 2
  @assert σ_r > 0 && σ_θ > 0 && σ₀_θ > 0

  knots_t = [0.0; sort(max.(eps(Float64), rand(rng, nknots))); 1.0]
  knots_dθ = [σ₀_θ * randn(rng, 1); σ_θ .* randn(rng, nknots + 1)]
  knots_r = abs.(1.0 .+ σ_r .* randn(rng, nknots + 2))

  t = 0.0:(1.0/(npoints-1)):1.0
  dθ = interpolate((knots_t,), knots_dθ, Gridded(Linear())).(t)
  r = interpolate((knots_t,), knots_r, Gridded(Linear())).(t)
  θ = cumsum(dθ)
  v = r .* [cos.(θ) sin.(θ)]
  cumsum(v, dims = 1)
end
