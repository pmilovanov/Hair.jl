using Flux
using Flux.Data: DataLoader


################################################################################
# GPU data loader
################################################################################

struct GPUDataLoader
  inner # iterable
end

function Base.iterate(d, i = 0)
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
    bufsize::Int = 256,
    shuffle::Bool = true,
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
      Channel(bufsize*4),
      Channel(bufsize),
    )
    @asynclog read_images_masks(this.c_blobs, filenames = filenames)
    @asynclog load_images_masks(this.c_blobs, this.c_imgs)

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
      X[:, :, :, j] = Float32.(permutedims(channelview(img), [2, 3, 1]))
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
