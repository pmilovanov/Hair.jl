using Base.Iterators: partition
using Flux.Data: DataLoader
using Parameters: @with_kw
using Flux, CUDA
using Flux: throttle, logitbinarycrossentropy
using Statistics
using Printf
using MLDataPattern: splitobs, shuffleobs
using ProgressMeter
using BSON: @load, @save
import Dates
using NNlib

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

@with_kw mutable struct TrainArgs
  img_dir::String = "."
  lr::Float64 = 3e-3
  throttle::Int = 1
  epochs::Int = 100
  batch_size = 32
  image_size = 512
  savepath::String = "./"
  test_set_ratio = 0.2
  previous_saved_model = nothing


  blocksizes = [2, 3, 3, 3]
end

readdir_nomasks(dirpath::String) =
  [x for x in readdir(dirpath, join = true) if !contains(x, "-mask")]

function prepare_data(args::TrainArgs, tracker=StatsTracker())
  filenames = readdir_nomasks(args.img_dir)

  train_fnames, test_fnames = splitobs(shuffleobs(filenames), at = (1 - args.test_set_ratio))

  trainset = GPUBufDataLoader(
    ImageAndMaskLoader(train_fnames; batchsize = args.batch_size, bufsize = args.batch_size * 128, id="imgloader_train", statstracker=tracker),
    2,
    id="gpudl_train",
    statstracker=tracker
  )
  testset = GPUBufDataLoader(
    ImageAndMaskLoader(test_fnames; batchsize = args.batch_size, bufsize = args.batch_size * 8, id="imgloader_test", statstracker=tracker),
    2,
    id="gpudl_test",
    statstracker=tracker
  )


  trainset, testset
end


function prepare_data_old(args::TrainArgs)
  X, Y = load_data(args.img_dir)
  X_train, Y_train, X_test, Y_test = test_train_split(X, Y, args.test_set_ratio)

  trainset =
    DataLoader((X_train, Y_train), batchsize = args.batch_size, shuffle = true, partial = false) |>
    GPUDataLoader
  testset =
    DataLoader((X_test, Y_test), batchsize = args.batch_size, shuffle = false, partial = true) |>
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

function build_model_simple(args::TrainArgs = TrainArgs())

  maxpool() = MaxPool((2, 2))

  # transpose conv, upsamples 2x
  upsample_conv(channels::Pair{Int,Int}) =
    Chain(Upsample(), Conv((5, 5), channels, relu, pad = SamePad(), stride = (1, 1)))


  convs3 = Chain(#  DebugPrintSize("conv0"),
    conv_block(args.blocksizes[1], (3, 3), 3 => 16), # DebugPrintSize("convs1"),
    maxpool(),
    conv_block(args.blocksizes[2], (3, 3), 16 => 24), # DebugPrintSize("convs2"),
    maxpool(),
    conv_block(args.blocksizes[3], (3, 3), 24 => 32),  #DebugPrintSize("convs3"),
  )

  convs4u1 = Chain(
    maxpool(),
    conv_block(args.blocksizes[4], (3, 3), 32 => 64),
    #   DebugPrintSize("convs4"),
    upsample_conv(64 => 32),
    BatchNorm(32),
    #    DebugPrintSize("convs4u1"),
  )

  stacked1u2 = Chain(
    SkipUnit(convs3, convs4u1),
    upsample_conv(64 => 16),
    BatchNorm(16),
    Upsample(),
    Conv((5, 5), 16 => 1, σ, pad = SamePad(), stride = (1, 1)),
    #    DebugPrintSize("stacked1u2"),
  )

  return stacked1u2

end


function testset_loss(loss, testset)
  losses = [loss(x, y) for (x, y) in testset]
  return sum(losses) / length(losses)
end


function train(args::Union{Nothing,TrainArgs}; kwargs...)
  if args == nothing
    args =
      TrainArgs(test_set_ratio = 0.05, img_dir = expanduser("~/data/hair/hairy/exp/full128_0120"))
  end

  tracker = StatsTracker()

  @info "Setting up data"
  trainset, testset = prepare_data(args, tracker)

  if args.previous_saved_model == nothing
    @info "Making model"
    m = build_model_simple(args) |> gpu
  else
    @info "Loading previous model"
    Core.eval(Main, :(import NNlib))
    @load args.previous_saved_model model
    m = model |> gpu
  end

  loss(x, y) = sum(Flux.Losses.binarycrossentropy(m(x), y))
  loss(t::Tuple{X,X} where {X}) = loss(t...)

  model_dir = joinpath(args.savepath, Dates.format(Dates.now(), "yyyymmdd-HHMM"))
  if !isdir(model_dir)
    mkdir(model_dir)
  end

  opt = ADAM(args.lr)
  @info("Training....")
  # Starting to train models
  f1_old = 0.0

  for i = 1:args.epochs
    p = Progress(length(trainset), dt = 1.0, desc = "Epoch $i: ")

    lasttime = Ref(time_ns())
    
    function iteration_callback()
      curtime = time_ns()
      elapsed = (curtime - lasttime[])/1e9
      report!(tracker, "train_iteration_time", elapsed)
      next!(p)
      lasttime[] = time_ns()
    end
    Flux.train!(loss, params(m), trainset, opt, cb = iteration_callback)
    #p,r,f1 = prf1(m, testset)
    #@printf("Epoch %3d PRF1: %0.3f   %0.3f   %0.3f   --- ", i, p, r, f1)

    p, r, f1 = prf1(m, testset)
    @printf("TEST  : P=%0.3f  R=%0.3f F1=%0.3f", p, r, f1)

    if f1 > f1_old
      modelfilename = joinpath(model_dir, @sprintf("epoch_%03d.bson", i))
      model = cpu(m)
      @save modelfilename model
      @info "Saved model to $modelfilename"
    end

    if mod(i, 5) == 0
      trainset = reset(trainset)
      p, r, f1 = prf1(m, trainset)
      @printf("TRAIN : P=%0.3f  R=%0.3f F1=%0.3f", p, r, f1)
    end
    #    @show f2

    stats = snapshot(tracker)
    for k in sort(collect(keys(stats)))
      println("----------- $k -------------")
      show(stats[k])
    end
    
    trainset, testset = reset(trainset), reset(testset)
  end

  return model_dir
end

# function main() end

# if abspath(PROGRAM_FILE) == @__FILE__
#    train()
# end

function infer_old(model, image::Image{RGB{Float32}}; inference_side::Int = 1024, device = gpu)
  model = device(model)
  newsize = convert.(Int, ceil.(size(image) ./ inference_side)) .* inference_side
  needresize = (size(image) != newsize)
  newimg = needresize ? imresize(image, newsize) : image

  step::Int = inference_side ÷ 2
  h = size(image)[1] / step - 1
  w = size(image)[2] / step - 1

  Ys = zeros(Float32, newsize...)

  for i = 1:h
    for j = 1:w
      (x0, x1) = convert.(Int, (i - 1, i + 1) .* step .+ (1, 0))
      (y0, y1) = convert.(Int, (j - 1, j + 1) .* step .+ (1, 0))

      X = device(imgtoarray(newimg[x0:x1, y0:y1]))
      Y = cpu(model(X))
      Ys[x0:x1, y0:y1] .= max.(Ys[x0:x1, y0:y1], Y[:, :, 1, 1])

      #Ys[x0:x1, y0:y1] .= Y[inference_side, inference_side, 1, 1]
    end
  end

  outimg = Gray{Float32}.(Ys)
  return needresize ? imresize(outimg, size(image)) : outimg
end

function infer(
  model,
  image::Image{RGB{Float32}};
  inference_side::Int = 1024,
  overlap = 512,
  batchsize = 4,
  device = gpu,
)
  model = device(model)

  step::Int = inference_side ÷ 2
  h = size(image)[1] / step - 1
  w = size(image)[2] / step - 1

  Ys = zeros(Float32, size(image)...)

  subimages = sample_image_w_coords(
    image,
    GridStrategy(side = inference_side, overlap = overlap, cover_edges = true),
  )

  while true
    done = false
    coords = zeros(Int, batchsize, 2)
    X = zeros(Float32, inference_side, inference_side, 3, batchsize)
    try
      for i = 1:batchsize
        subimage = pop!(subimages)
        coords[i, :] .= first(subimage)
        X[:, :, :, i] .= imgtoarray(last(subimage))[:, :, :, 1]
      end
    catch e
      @assert isa(e, ArgumentError)
      done = true
    end

    X = X |> device
    Y = model(X) |> cpu

    for i = 1:batchsize
      x0, y0 = coords[i, :]
      if (x0, y0) == (0, 0)
        break
      end
      x1, y1 = (x0, y0) .+ inference_side .- 1
      Ys[x0:x1, y0:y1] .= max.(Ys[x0:x1, y0:y1], Y[:, :, 1, i])
    end

    if done
      break
    end
  end

  return Gray{Float32}.(Ys)
end



function infer(model, path::String; kwargs...)
  img = RGB{Float32}.(load(path))
  (img, infer(model, img; kwargs...))
end

function infer_on_image(model, path::String, side::Int = 1024, mask_threshold = -1.0; device = gpu)
  model = device(model)

  img = load(path)
  imgx = imgtoarray(img, side) |> device
  y = cpu(model(imgx))[:, :, 1, 1]
  if mask_threshold >= 0
    y = y .> mask_threshold
  end
  imgy = Gray{N0f8}.(y)
  return (img, imresize(imgy, size(img)...))
end

function place_overlay!(
  img::Image{T} where {T},
  overlay::Image{Gray{G}} where {G},
  mask_threshold = 0.1,
  outline = true,
)
  rimg = rawview(channelview(img))

  if outline == true
    binmask = overlay .> mask_threshold
    for i = 1:7
      dilate!(binmask)
    end

    outline = copy(binmask)
    for i = 1:3
      dilate!(outline)
    end

    outline = outline .* (.!binmask)
    overlay = convert.(eltype(overlay), outline)
  end

  roverlay = rawview(channelview(overlay))
  rimg[1, :, :] = max.(rimg[1, :, :], roverlay)

  img
end


function infer_overlay(model, path::String, side::Int = 1024, mask_threshold = 0.1; kwargs...)
  img, mask = infer(model, path; kwargs...)
  @show size(mask)
  @show size(img)
  mask = Gray.(mask .> mask_threshold)

  overlay = place_overlay!(copy(img), mask)
  return (img, overlay)
end



function latestmodel(; dir = expanduser("~/data/hair/models"), modelid = nothing, epoch = 0)
  modeldir =
    (modelid == nothing) ? sort(readdir(expanduser(dir), join = true), rev = true)[1] :
    joinpath(dir, modelid)
  return (epoch == 0) ? sort(readdir(modeldir, join = true), rev = true)[1] :
         joinpath(modeldir, @sprintf("epoch_%03d.bson", epoch))
end
