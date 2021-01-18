using Base.Iterators: partition
using Flux.Data: DataLoader
using Parameters: @with_kw
using Flux, CUDA
using Flux: throttle, binarycrossentropy
if has_cuda()
  @info "CUDA is on"
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

@with_kw mutable struct TrainArgs
  img_dir::String = "."
  lr::Float64 = 3e-3
  throttle::Int = 1
  epochs::Int = 20
  batch_size = 4
  image_size = 512
  savepath::String = "./"
  test_set_ratio = 0.01
end

function prepare_data(args::TrainArgs)
  X, Y = load_data(args.img_dir)
  X_train, Y_train, X_test, Y_test = test_train_split(X, Y, args.test_set_ratio)

  trainset =
    DataLoader((X_train, Y_train), batchsize = args.batch_size, shuffle = true, partial = false) |>
    GPUDataLoader
  testset =
    DataLoader((X_test, Y_test), batchsize = args.batch_size, shuffle = false, partial = false) |>
    GPUDataLoader

  trainset, testset
end

function conv_block(
  nunits::Int,
  k::Tuple{Int,Int},
  ch::Pair{<:Integer,<:Integer},
  σ = relu;
  pad = (1, 1),
  stride = (1, 1),
  kwargs...,
) where {N}
  chain = [Conv(k, ch, σ; pad = pad, stride = stride, kwargs...), BatchNorm(last(ch))]
  for i = 1:nunits-1
    push!(chain, Conv(k, last(ch) => last(ch), σ; pad = pad, stride = stride, kwargs...))
    push!(chain, BatchNorm(last(ch)))
  end
  Chain(chain...)
end

function build_model(args::TrainArgs = TrainArgs())

  maxpool() = MaxPool((2, 2))

  convs3 = Chain( # DebugPrintSize("conv0"),
    conv_block(2, (3, 3), 3 => 64), # DebugPrintSize("convs1"),
    maxpool(),
    conv_block(3, (3, 3), 64 => 128), # DebugPrintSize("convs2"),
    maxpool(),
    conv_block(3, (3, 3), 128 => 256), # DebugPrintSize("convs3"),
  )

  convs5 = Chain(
    convs3,
    maxpool(),
    conv_block(3, (3, 3), 256 => 512), # DebugPrintSize("convs4"),
    maxpool(),
    conv_block(3, (3, 3), 512 => 512), # DebugPrintSize("convs5"),
  )

  # transpose conv, upsamples 2x
  upsample_conv(channels::Pair{Int,Int}) =
    Chain(Upsample(), Conv((3, 3), channels, relu, pad = SamePad(), stride = (1, 1)))

  convs5u2 = Chain(
    convs5,
    upsample_conv(512 => 256),
    BatchNorm(256),
    upsample_conv(256 => 256),
    BatchNorm(256),
  )

  stacked1u2 = Chain(
    StackChannels(convs5u2, convs3),
    upsample_conv(512 => 64),
    BatchNorm(256),
    upsample_conv(64 => 1),
  )

  return stacked1u2

end

function train(; kwargs...)
  args = TrainArgs()

  args.img_dir = expanduser("~/data/hair/hairy/exp/0113-01")

  @info "Loading data"
  trainset, testset = prepare_data(args)

  @info "Making model"
  m = build_model(args) |> gpu

  loss(x, y) = Flux.binarycrossentropy(m(x), y)

  evalcb = throttle(() -> @show(loss(testset...)), args.throttle)
  opt = ADAM(args.lr)
  @info("Training....")
  # Starting to train models
  #Flux.@epochs args.epochs
  Flux.train!(loss, params(m), trainset, opt, cb = evalcb)
end

# function main() end

# if abspath(PROGRAM_FILE) == @__FILE__
#    train()
# end
