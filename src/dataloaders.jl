using Flux
using Flux.Data: DataLoader


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


################################################################################

FilenameImageMaskTuple{P,Q} = Tuple{String,Array{P,2},Array{Q,2}} where {P,Q}
struct ImageAndMaskLoader
  filenames::Array{String,1}
  batchsize::Int
  bufsize::Int
  imgsize::Tuple{Int,Int}
  imgnumchannels::Int

  c_blobs::Channel{Array{UInt8,1}}
  c_imgs::Channel{Array{FilenameImageMaskTuple,1}}

  function ImageAndMaskLoader(
    filenames::Array{String,1};
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
      size(sampleimg),
      length(sampleimg[1, 1]),
      Channel{Array{UInt8,1}}(bufsize),
      Channel{Array{FilenameImageMaskTuple,1}}(bufsize),
    )
    @asynclog read_images_masks(this.c_blobs, filenames)
    @asynclog load_images_masks(this.c_blobs, this.c_imgs)
  end
end

function Base.iterate(d::ImageAndMaskLoader, i = 0)
  X = zeros(Float32, d.imgsize..., d.imgnumchannels, d.batchsize)
  Y = zeros(Float32, d.imgsize..., 1, d.batchsize)

  for j = 1:d.batchsize
    try
      (fname, img, mask) = take!(d.c_imgs)
      X[:, :, :, j] = Float32.(permutedims(channelview(img), [2, 3, 1]))
      Y[:, :, 1, j] = Float32.(mask .> 0.9)
    catch e
      if isopen(d.c_imgs)
        rethrow(e)
      end
      return nothing
    end
  end

  return (X, Y), i + 1
end
