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


stack_channels(x1::AbstractArray, x2::AbstractArray) = cat(x1, x2, dims = 3)
struct SkipUnit{P,Q,F}
  trunk::P
  branch::Q
  combine_fn::F
end
SkipUnit(trunk, branch) = SkipUnit(trunk, branch, stack_channels)

Flux.@functor SkipUnit
function (s::SkipUnit)(x::AbstractArray)
  y_trunk = s.trunk(x)
  y_branch = s.branch(y_trunk)
  return s.combine_fn(y_trunk, y_branch)
end


"""
Passthrough debugging layer that prints size of its input.
"""
struct DebugPrintSize
  name::AbstractString
end
Flux.@functor DebugPrintSize
function (f::DebugPrintSize)(x::AbstractArray)
  println("Output $(f.name): $(size(x))")
  x
end


################################################################################
## Metrics
################################################################################

binfloat(x::AbstractArray{T,N}) where {T<:AbstractFloat,N} =
  clamp.(round.(x), convert(T, 0), convert(T, 1))

count1s(x::AbstractArray{T,N}) where {T<:AbstractFloat,N} = sum(binfloat(x))


struct BinarySegmentationMetrics
  precision::Float64
  recall::Float64
  f1::Float64
  ap_ŷ::Int
  ap_y::Int
  tp::Int
  tn::Int
  fp::Int
  fn::Int
  npixels::Int
  loss::Float64
end

BinarySegmentationMetrics() =
  BinarySegmentationMetrics(zeros(Int, fieldcount(BinarySegmentationMetrics))...)
function totuple(z::BinarySegmentationMetrics)
  fnames = fieldnames(BinarySegmentationMetrics)
  [getfield(z, x) for x in fnames]
end


function binsegmetrics(
  ŷ::AbstractArray{Bool,N},
  y::AbstractArray{Bool,N};
  lossfn::Union{Function,Nothing} = nothing,
) where {N}
  @assert size(ŷ) == size(y)
  npixels = prod(size(y))
  ap_ŷ = count(ŷ)
  ap_y = count(y)
  tp = count(ŷ .& y)
  fp = count(ŷ) - tp

  nŷ, ny = .!ŷ, .!y
  tn = count(nŷ .& ny)
  fn = count(nŷ) - tn

  p = tp / (tp + fp)
  r = tp / (tp + fn)
  f1 = 2 * p * r / (p + r)

  loss = isnothing(lossfn) ? -1.0 : lossfn(ŷ, y)

  BinarySegmentationMetrics(p, r, f1, ap_ŷ, ap_y, tp, tn, fp, fn, npixels, loss)
end


function binsegmetrics(
  ŷ::AbstractArray{T,N},
  y::AbstractArray{T,N},
  threshold::AbstractFloat = 0.1;
  kwargs...,
) where {T<:AbstractFloat,N}
  ŷ = (ŷ .> threshold)
  y = (y .> threshold)
  binsegmetrics(ŷ, y; kwargs...)
end

function precision(ŷ::AbstractArray{T,N}, y::AbstractArray{T,N}) where {T<:AbstractFloat,N}
  ŷr, yr = binfloat(ŷ), binfloat(y)
  return sum(ŷr .* yr) / sum(ŷr)
end

function recall(ŷ::AbstractArray{T,N}, y::AbstractArray{T,N}) where {T<:AbstractFloat,N}
  ŷr, yr = binfloat(ŷ), binfloat(y)
  return sum(ŷr .* yr) / sum(yr)
end

function f1(ŷ::AbstractArray{T,N}, y::AbstractArray{T,N}) where {T<:AbstractFloat,N}
  p += precision(ŷ, y)
  r += recall(ŷ, y)
  f1 = 2 * p * r / (p + r)
end

function prf1(ŷ::AbstractArray{T,N}, y::AbstractArray{T,N}) where {T<:AbstractFloat,N}
  p = precision(ŷ, y)
  r = recall(ŷ, y)
  f1 = 2 * p * r / (p + r)
  (p, r, f1, 0.0)
end

precision(model, testset) = mean([precision(model(x), y) for (x, y) in testset])
recall(model, testset) = mean([recall(model(x), y) for (x, y) in testset])


function prf1(model, testset; lossfn::Union{Function,Nothing} = nothing)
  p, r, loss = 0.0, 0.0, 0.0
  n = 0

  for (x, y) in testset
    ŷ = model(x)
    metrics = binsegmetrics(ŷ, y, lossfn = lossfn)

    p += metrics.precision
    r += metrics.recall
    n += 1
    loss += metrics.loss
  end
  p, r, loss = p / n, r / n, loss / n
  f1 = 2 * p * r / (p + r)
  return (p = p, r = r, f1 = f1, loss = loss)
end
