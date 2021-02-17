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
import JSON3

using .Models

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
  only_save_model_if_better = true
end

readdir_nomasks(dirpath::String) =
  [x for x in readdir(dirpath, join = true) if !contains(x, "-mask")]

function prepare_data(args::TrainArgs, tracker = StatsTracker())
  filenames = readdir_nomasks(args.img_dir)

  train_fnames, test_fnames = splitobs(shuffleobs(filenames), at = (1 - args.test_set_ratio))

  # trainset = GPUBufDataLoader(
  #   ImageAndMaskLoader(
  #     train_fnames;
  #     batchsize = args.batch_size,
  #     bufsize = args.batch_size * 128,
  #     id = "imgloader_train",
  #     statstracker = tracker,
  #   ),
  #   2,
  #   id = "gpudl_train",
  #   statstracker = tracker,
  # )
  # testset = GPUBufDataLoader(
  #   ImageAndMaskLoader(
  #     test_fnames;
  #     batchsize = args.batch_size,
  #     bufsize = args.batch_size * 8,
  #     id = "imgloader_test",
  #     statstracker = tracker,
  #   ),
  #   2,
  #   id = "gpudl_test",
  #   statstracker = tracker,
  # )

  trainset =
    ImageAndMaskLoader(
      train_fnames;
      batchsize = args.batch_size,
      bufsize = args.batch_size * 16,
      id = "imgloader_train",
      statstracker = tracker,
    ) |> GPUDataLoader

  testset =
    ImageAndMaskLoader(
      test_fnames;
      batchsize = args.batch_size,
      bufsize = args.batch_size * 8,
      id = "imgloader_test",
      statstracker = tracker,
    ) |> GPUDataLoader

  trainset, testset
end



function testset_loss(loss, testset)
  losses = [loss(x, y) for (x, y) in testset]
  return sum(losses) / length(losses)
end


bce_loss(model) = (x, y) -> sum(Flux.Losses.binarycrossentropy(model(x), y))
bce_loss_tuple(model) = xy -> bce_loss(model)(xy...)


function train(args::TrainArgs, am::Union{Models.AnnotatedModel,Nothing} = nothing; kwargs...)
  @info "Number of threads: $(Threads.nthreads())"
  tracker = StatsTracker()

  @info "Setting up data"
  trainset, testset = prepare_data(args, tracker)

  Models.setmeta!(am, :train_args, args)

  if args.previous_saved_model == nothing
    if am == nothing
      throw(ArgumentError("model must not be nothing if args.previous_saved_model is not set"))
    end
    m = Models.model(am) |> gpu
  else
    @info "Loading previous model"
    Core.eval(Main, :(import NNlib))
    @load args.previous_saved_model model
    m = model |> gpu
  end


  model_dir = joinpath(args.savepath, Dates.format(Dates.now(), "yyyymmdd-HHMM"))
  isdir(args.savepath) || mkdir(args.savepath)
  isdir(model_dir) || mkdir(model_dir)

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
    Flux.train!(bce_loss(m), params(m), trainset, opt, cb = iteration_callback)
    #p,r,f1 = prf1(m, testset)
    #@printf("Epoch %3d PRF1: %0.3f   %0.3f   %0.3f   --- ", i, p, r, f1)

    p, r, f1 = prf1(m, testset)
    Models.setmeta!(am, :metrics, Dict(:p => p, :r => r, :f1 => f1))
    @info @sprintf("TEST  : P=%0.3f R=%0.3f F1=%0.3f", p, r, f1)

    if args.only_save_model_if_better == false || f1 > f1_old
      modelfilename = joinpath(model_dir, @sprintf("epoch_%03d.bson", i))
      metafilename = joinpath(model_dir, @sprintf("epoch_%03d.json", i))
      model = cpu(m)
      @save modelfilename model
      Models.savemeta(am, metafilename)

      @info "Saved model to $modelfilename"
    end

    if mod(i, 5) == 0
      trainset = reset(trainset)
      p, r, f1 = prf1(m, trainset)
      @info @sprintf("TRAIN : P=%0.3f R=%0.3f F1=%0.3f", p, r, f1)
    end
    #    @show f2


    # stats = snapshot(tracker)
    # for k in sort(collect(keys(stats)))
    #   println("----------- $k -------------")
    #   show(stats[k])
    # end

    trainset, testset = reset(trainset), reset(testset)
  end

  return model_dir
end
