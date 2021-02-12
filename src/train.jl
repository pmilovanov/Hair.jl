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
  @spawnlog c_blobs read_images_masks_from_dir(c_blobs, datadir)
  @spawnlog c_imgs load_images_masks(c_blobs, c_imgs)

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

function prepare_data(args::TrainArgs, tracker = StatsTracker())
  filenames = readdir_nomasks(args.img_dir)

  train_fnames, test_fnames = splitobs(shuffleobs(filenames), at = (1 - args.test_set_ratio))

  trainset = GPUBufDataLoader(
    ImageAndMaskLoader(
      train_fnames;
      batchsize = args.batch_size,
      bufsize = args.batch_size * 128,
      id = "imgloader_train",
      statstracker = tracker,
    ),
    2,
    id = "gpudl_train",
    statstracker = tracker,
  )
  testset = GPUBufDataLoader(
    ImageAndMaskLoader(
      test_fnames;
      batchsize = args.batch_size,
      bufsize = args.batch_size * 8,
      id = "imgloader_test",
      statstracker = tracker,
    ),
    2,
    id = "gpudl_test",
    statstracker = tracker,
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
    m = Models.build_model_simple(args.blocksizes) |> gpu
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
      elapsed = (curtime - lasttime[]) / 1e9
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
