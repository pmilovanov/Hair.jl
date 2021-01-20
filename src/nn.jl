using Flux
using Flux.Data: DataLoader
using Parameters: @with_kw
using Zygote
using CUDA

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

nminibatches(X::Array{T,4}, batchsize::Int) where {T} = batchsize(X)[4] ÷ size

function minibatch(batchid::Int, X::Array{T,4}, Y::Array{T,4}, batchsize::Int) where {T}
  @assert batchid > 0
  @assert batchid <= nminibatches(X, batchsize)
  a, b = (batchid - 1) * batchsize + 1, batchid * batchsize
  return @view(X[:, :, :, a:b]), @view(Y[:, :, :, a:b])
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
function (u::Upsample)(x::AbstractArray{Float32, N}) where N
  ratio = u.ratio
  (h, w, c, n) = size(x)
  y = fill(1.0f0, (ratio[1], 1, ratio[2], 1, 1, 1))
  z = reshape(x, (1, h, 1, w, c, n)) .* y
  reshape(z, size(x) .* ratio)
end
function (u::Upsample)(x::CuArray{Float32, N}) where N
  ratio = u.ratio
  (h, w, c, n) = size(x)
  y = CUDA.fill(1.0f0, (ratio[1], 1, ratio[2], 1, 1, 1))
  z = CUDA.reshape(x, (1, h, 1, w, c, n)) .* y
  CUDA.reshape(z, size(x) .* ratio)
end

Zygote.@adjoint CUDA.fill(x::Real, dims...) = CUDA.fill(x, dims...), Δ->(sum(Δ), map(_->nothing, dims)...)

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
# GPU data loader
################################################################################


struct GPUDataLoader{D}
  inner::DataLoader{D}
end

function Base.iterate(d::GPUDataLoader, i = 0)
  it = Base.iterate(d.inner, i)
  if it == nothing
    return nothing
  else
    data, nexti = it
    return (gpu(data), nexti)
  end
end
Base.length(d::GPUDataLoader) = Base.length(d.inner)

Base.eltype(::GPUDataLoader{D}) where {D} = D
