using Flux
using Flux.Data: DataLoader
using Parameters: @with_kw
using Zygote
using CUDA
using Statistics

function test_train_split(
  X::AbstractArray{T,4},
  Y::AbstractArray{T,4},
  test_ratio::Float64 = 0.05,
) where {T}
  @assert size(X)[4] == size(Y)[4]
  len = size(X)[4]
  boundary = len - convert(Int, ceil(test_ratio * Float64(len)))
  X_train, Y_train = @view(X[:, :, :, 1:boundary]), @view(Y[:, :, :, 1:boundary])
  X_test, Y_test = @view(X[:, :, :, boundary+1:len]), @view(Y[:, :, :, boundary+1:len])
  X_train, Y_train, X_test, Y_test
end


################################################################################
## Layers
################################################################################


# Stolen from https://discourse.julialang.org/t/upsampling-in-flux-jl/25919/4
"""
Differentiable layer to upsample a tensor.
"""
@with_kw struct Upsample
  ratio::Tuple{Int,Int,Int,Int} = (2, 2, 1, 1)
end
Flux.@functor Upsample
function (u::Upsample)(x::AbstractArray{Float32,N}) where {N}
  ratio = u.ratio
  (h, w, c, n) = size(x)
  y = fill(1.0f0, (ratio[1], 1, ratio[2], 1, 1, 1))
  z = reshape(x, (1, h, 1, w, c, n)) .* y
  reshape(z, size(x) .* ratio)
end
function (u::Upsample)(x::CuArray{Float32,N}) where {N}
  ratio = u.ratio
  (h, w, c, n) = size(x)
  y = CUDA.fill(1.0f0, (ratio[1], 1, ratio[2], 1, 1, 1))
  z = CUDA.reshape(x, (1, h, 1, w, c, n)) .* y
  CUDA.reshape(z, size(x) .* ratio)
end

Zygote.@adjoint CUDA.fill(x::Real, dims...) =
  CUDA.fill(x, dims...), Δ -> (sum(Δ), map(_ -> nothing, dims)...)

"""
Layer to concat outputs of two layers along the channel dimension.
Layers are size (h, w, nchannels, batches)
"""
struct StackChannels{P,Q}
  layer1::P
  layer2::Q
end
Flux.@functor StackChannels
function (s::StackChannels)(x::AbstractArray)
  y1 = s.layer1(x)
  y2 = s.layer2(x)
  cat(y1, y2, dims = 3)
end

"""
Passthrough debugging layer that prints size of its input.
"""
struct DebugPrintSize
  name::AbstractString
end
Flux.@functor DebugPrintSize
function (f::DebugPrintSize)(x::AbstractArray)
  @info "Output $(f.name): $(size(x))"
  x
end


################################################################################
## Metrics
################################################################################

binfloat(x::AbstractArray{T,N}) where {T<:AbstractFloat,N} =
  clamp.(round.(x), convert(T, 0), convert(T, 1))

count1s(x::AbstractArray{T,N}) where {T<:AbstractFloat,N} =
  sum(binfloat(x))

function precision(ŷ::AbstractArray{T,N}, y::AbstractArray{T,N}) where {T<:AbstractFloat,N}
  ŷr, yr = binfloat(ŷ), binfloat(y)
  return sum(ŷr .* yr) / sum(ŷr)
end

function recall(ŷ::AbstractArray{T,N}, y::AbstractArray{T,N}) where {T<:AbstractFloat,N}
  ŷr, yr = binfloat(ŷ), binfloat(y)
  return sum(ŷr .* yr) / sum(yr)
end

precision(model, testset) = mean([precision(model(x), y) for (x,y) in testset])
recall(model, testset) = mean([recall(model(x), y) for (x,y) in testset])

function prf1(model, testset)
  p, r = 0.0, 0.0
  n = 0
  for (x,y) in testset
    ŷ = model(x)
    p += precision(ŷ,y)
    r += recall(ŷ,y)
    n += 1
  end
  p, r = p/n, r/n
  f1 = 2*p*r/(p+r)
  return p, r, f1
end
