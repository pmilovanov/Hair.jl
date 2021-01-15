using Base.Iterators: partition
using Flux.Data: DataLoader
using Parameters: @with_kw
using Flux, CUDA
if has_cuda()
  @info "CUDA is on"
  CUDA.allowscalar(false)
end

function load_data(datadir::String; bufsize::Int = 256)
  fnames = [x for x in readdir(datadir, join = true) if !contains(x, "-mask")]
  nimgs = length(fnames)
  @assert !contains(fnames[1], "-mask")
  w = size(load(fnames[1]))[1]

  imgs = Array{Float32,4}(undef, w, w, 3, nimgs)
  masks = Array{Float32,4}(undef, w, w, 1, nimgs)

  c_blobs = Channel(bufsize)
  c_imgs = Channel(bufsize)
  @asynclog read_images_masks_from_dir(c_blobs, datadir)
  @asynclog load_images_masks(c_blobs, c_imgs)

  for (i, (fname, img, mask)) in enumerate(c_imgs)
    imgs[:, :, :, i] = Float32.(permutedims(channelview(img), [2, 3, 1]))
    masks[:, :, 1, i] = Float32.(mask .> 0.9)
  end

  imgs, masks
end

nminibatches(X::Array{T,4}, batchsize::Int) where {T} = batchsize(X)[4] ÷ size


function minibatch(batchid::Int, X::Array{T,4}, Y::Array{T,4}, batchsize::Int) where {T}
  @assert batchid > 0
  @assert batchid <= nminibatches(X, batchsize)
  a, b = (batchid - 1) * batchsize + 1, batchid * batchsize
  return @view(X[:, :, :, a:b]), @view(Y[:, :, :, a:b])
end

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

@with_kw struct TrainArgs
  img_dir::String = "."
  lr::Float64 = 3e-3
  epochs::Int = 20
  batch_size = 128
  image_size = 512
  savepath::String = "./"
  test_set_ratio = 0.05
end

function prepare_data(args::TrainArgs)
  X, Y = gpu.(load_data(args.img_dir))
  X_train, Y_train, X_test, Y_test = test_train_split(X, Y, args.test_set_ratio)

  trainset =
    DataLoader((X_train, Y_train), batchsize = args.batch_size, shuffle = true, partial = false)
  testset =
    DataLoader((X_test, Y_test), batchsize = args.batch_size, shuffle = false, partial = false)

  trainset, testset
end



function conv_block(
  nunits::Int,
  k::Tuple{Int,Int},
  ch::Pair{<:Integer,<:Integer},
  σ = relu;
  pad=SamePad(),
  stride=1,
  kwargs...,
) where {N}
  chain = [Conv(k, ch, σ; kwargs...), BatchNorm(last(ch))]
  for i = 1:nunits-1
    push!(chain, Conv(k, last(ch) => last(ch), σ; pad=pad, stride=stride, kwargs...))
    push!(chain, BatchNorm(last(ch)))
  end
  Chain(chain...)
end




# Stolen from https://discourse.julialang.org/t/upsampling-in-flux-jl/25919/4
# struct Upsample end

# function upsample(x)
#   ratio = (2, 2, 1, 1)
#   (h, w, c, n) = size(x)
#   y = similar(x, (ratio[1], 1, ratio[2], 1, 1, 1))
#   fill!(y, 1)
#   z = reshape(x, (1, h, 1, w, c, n)) .* y
#   reshape(z, size(x) .* ratio)
# end

struct StackChannels{P,Q}
  layer1::P
  layer2::Q
end
Flux.@functor StackChannels
function (s::StackChannels)(x::AbstractArray)
  y1 = s.layer1(x)
  y2 = s.layer2(x)
  @show size(y1)
  @show size(y2)
  cat(y1, y2, dims=3)
end

function build_model(args::TrainArgs)

  maxpool = MaxPool((2, 2))
  convs1 = conv_block(2, (3, 3), 3 => 64, relu, pad = SamePad(), stride = 1)
  convs2 =
    Chain(convs1, maxpool, conv_block(2, (3, 3), 64 => 128))

  convs3 =
    Chain(convs2, maxpool, conv_block(3, (3, 3), 128 => 256))

  convs4 =
    Chain(convs3, maxpool, conv_block(3, (3, 3), 256 => 512))

  convs5 =
    Chain(convs4, maxpool, conv_block(3, (3, 3), 512 => 512))

  # transpose conv, upsamples 2x
  tconv(channels::Pair{Int,Int}) = Chain(
    ConvTranspose((3, 3), channels, relu, pad = SamePad(), stride = 2),
    BatchNorm(last(channels)),
  )

  convs5u2 = Chain(
    convs5,
    tconv(512 => 256),
    tconv(256 => 256)
  )
  
  stacked1u2 = Chain(StackChannels(convs5u2, convs3),
                     tconv(256=>64),
                     ConvTranspose((3, 3), 64=>1, σ, pad = SamePad(), stride = 2))

  return stacked1u2
  #  convs3u2 = Chain(convs3, upsample, upsample)

  #  Chain(x->cat(convs5u4()
end


function main() end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end
