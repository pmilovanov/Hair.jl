using Flux
using Flux.Data: DataLoader
using Images


################################################################################
# GPU data loader
################################################################################

"Wrapper around an iterable that moves data to the GPU before returning it to caller"
struct GPUDataLoader
  inner::Any # iterable
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

reset(d::GPUDataLoader) = GPUDataLoader(reset(d.inner))


"""
Buffered wrapper around an iterable that asynchronously moves data to the GPU before handing it to the consumer.

This means that it aims to pre-load some data into GPU memory so that it's there when the consumer wants to use it.
How much data is preloaded is determined by the size of the buffer.

Also takes a StatsTracker to track channel performance.
"""
struct GPUBufDataLoader
  inner::Any # iterable

  bufsize::Int
  chan::Channel

  function GPUBufDataLoader(inner_loader, bufsize::Int; id = "gpudl", statstracker = StatsTracker())
    chan = Channel(bufsize)
    this = new(inner_loader, bufsize, chan)
    @spawnlog this.chan for data in this.inner
      put!(this.chan, gpu(data))
    end
    this
  end
end

function Base.iterate(d::GPUBufDataLoader, i = 0)
  try
    return (take!(d.chan), i + 1)
  catch e
    if isopen(d.chan)
      rethrow(e)
    end
    return nothing
  end
end
Base.length(d::GPUBufDataLoader) = Base.length(d.inner)

function reset(d::GPUBufDataLoader)
  if isopen(d.chan)
    close(d.chan)
  end
  return GPUBufDataLoader(reset(d.inner), d.bufsize)
end



################################################################################

FilenameImageMaskTuple{P,Q} = Tuple{String,Array{P,2},Array{Q,2}} where {P,Q}


struct ImageAndMaskLoader
  filenames::AbstractArray{String,1}
  batchsize::Int
  bufsize::Int
  shuffle::Bool
  imgsize::Tuple{Int,Int}
  imgnumchannels::Int

  c_blobs::Channel #{Array{UInt8,1}}
  c_imgs::Channel #{Array{FilenameImageMaskTuple,1}}

  function ImageAndMaskLoader(
    filenames::AbstractArray{String,1};
    batchsize::Int = 32,
    bufsize::Int = 64,
    shuffle::Bool = true,
    id = "imgloader",
    statstracker = StatsTracker(),
  )
    @assert length([x for x in filenames if contains(x, "-mask")]) == 0

    sampleimg = load(filenames[1])

    if shuffle
      Random.shuffle!(filenames)
    end

    this = new(
      filenames,
      batchsize,
      bufsize,
      shuffle,
      size(sampleimg),
      3,
      Channel(bufsize),
      Channel(bufsize)
    )
    @spawnlog this.c_blobs read_images_masks(this.c_blobs, filenames = filenames)
    @spawnlog this.c_imgs load_images_masks(this.c_blobs, this.c_imgs)

    return this
  end
end

reset(d::ImageAndMaskLoader) =
  ImageAndMaskLoader(d.filenames; batchsize = d.batchsize, bufsize = d.bufsize, shuffle = d.shuffle)

function Base.iterate(d::ImageAndMaskLoader, i = 0)
  X = zeros(Float32, d.imgsize..., d.imgnumchannels, d.batchsize)
  Y = zeros(Float32, d.imgsize..., 1, d.batchsize)

  for j = 1:d.batchsize
    try
      (fname, img, mask) = take!(d.c_imgs)
      # @show typeof(mask)
      # @show eltype(mask)
      X[:, :, :, j] = permuteddimsview(channelview(img), [2, 3, 1])
      Y[:, :, 1, j] = Float32.(mask .> 0.9)
    catch e
      if isopen(d.c_imgs)
        rethrow(e)
      end
      return nothing
    end
  end

  return ((X, Y), i + 1)
end

Base.length(d::ImageAndMaskLoader) = length(d.filenames) ÷ d.batchsize



imgtoarray(img::Image) = Float32.(Flux.unsqueeze(permutedims(channelview(img), [2, 3, 1]), 4))

function imgtoarray(img::Image, side::Int)
  if size(img) != (side, side)
    img = imresize(img, side, side)
  end
  imgtoarray(img)
end

arraytoimg(arr::AbstractArray{T,3}) where {T} = colorview(RGB, permuteddimsview(arr, (3, 1, 2)))

################################################################################

mutable struct SyncIMLoader
  filenames::AbstractArray{String,1}
  batchsize::Int
  shuffle::Bool
  imgsize::Tuple{Int,Int}
  imgnumchannels::Int

  function SyncIMLoader(
    filenames::AbstractArray{String,1};
    batchsize::Int = 32,
    shuffle::Bool = true
  )
    @assert length([x for x in filenames if contains(x, "-mask")]) == 0

    sampleimg = load(filenames[1])

    if shuffle
      Random.shuffle!(filenames)
    end

    return new(
      filenames,
      batchsize,
      shuffle,
      size(sampleimg),
      3
    )
  end
end

reset(d::SyncIMLoader) =
  SyncIMLoader(d.filenames; batchsize = d.batchsize, shuffle = d.shuffle)

function Base.iterate(d::SyncIMLoader, i = 1)
  X = zeros(Float32, d.imgsize..., d.imgnumchannels, d.batchsize)
  Y = zeros(Float32, d.imgsize..., 1, d.batchsize)

  if i > length(d); return nothing; end
  
  for j = 1:d.batchsize
      img = convert.(RGB{Float32}, load(d.filenames[j+i-1]))
      mask = convert.(Gray{Float32}, load(maskfname(d.filenames[j+i-1])))
      
      # @show typeof(mask)
      # @show eltype(mask)
      X[:, :, :, j] = permuteddimsview(channelview(img), [2, 3, 1])
      Y[:, :, 1, j] = Float32.(mask .> 0.9)
  end

  return ((X, Y), i + 1)
end

Base.length(d::SyncIMLoader) = length(d.filenames) ÷ d.batchsize
