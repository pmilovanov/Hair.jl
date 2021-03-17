import Dates
import JSON3
using BSON: @load, @save
using Base.Iterators: partition
using Flux, CUDA
using Flux.Data: DataLoader
using Flux: throttle, logitbinarycrossentropy
using Logging, LoggingExtras
using MLDataPattern: splitobs, shuffleobs
using NNlib
using Parameters: @with_kw
using Printf
using ProgressMeter
using Statistics
using StatsBase
using TensorBoardLogger


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
  modeldir::String = "./"
  test_set_ratio = 0.2
  only_save_model_if_better = true
end

readdir_nomasks(dirpath::String) =
  [x for x in readdir(dirpath, join = true) if !contains(x, "-mask")]

function maybe_download_data(path::String)
  if !isgcs(path); return path; end

  path = GCSPath(path)
  tdir = mktempdir(cleanup=false)
  @info "Downloading data and untarring to dir $(tdir)"
  untar(path, tdir)
  tdir
end

function prepare_data(args::TrainArgs, tracker = StatsTracker())

  img_dir = maybe_download_data(args.img_dir)

  filenames = readdir_nomasks(img_dir)

  train_fnames, test_fnames = splitobs(shuffleobs(filenames), at = (1 - args.test_set_ratio))
  train_fnames_subsample = sample(train_fnames, length(test_fnames), replace=false)

  trainset =
    SegmentationDataLoader(
      train_fnames;
      batchsize = args.batch_size
    ) |> GPUDataLoader

  trainset_subsample =
    SegmentationDataLoader(
      train_fnames_subsample;
      batchsize = args.batch_size
    ) |> GPUDataLoader

  testset =
    SegmentationDataLoader(
      test_fnames;
      batchsize = args.batch_size
    ) |> GPUDataLoader

  trainset, testset, trainset_subsample
end



function testset_loss(loss, testset)
  losses = [loss(x, y) for (x, y) in testset]
  return sum(losses) / length(losses)
end


# function savemodel(model, am::Models.AnnotatedModel)
# modelfilename = joinpath(model_dir, @sprintf("epoch_%03d.bson", i))
#       metafilename = joinpath(model_dir, @sprintf("epoch_%03d.json", i))
#       model = cpu(m)
#       @save modelfilename model
#       Models.savemeta(am, metafilename)
#                    @info "Saved model to $modelfilename"
#                    end

bce_loss(model) = (x, y) -> sum(Flux.Losses.binarycrossentropy(model(x), y))
bce_loss_tuple(model) = xy -> bce_loss(model)(xy...)

function loadmodel(path::String)
    @info "Loading previous model"
    Core.eval(Main, :(import NNlib))
    Core.eval(Main, :(import Flux))
    @load downloadmemaybe(path) model
    model
end

function maybeloadmodel(makemodelfn::Function, modeldir::String; prefix="epoch_", ext=".bson")
  """
  Load the last model checkpoint from disk or make a new one using `makemodelfn`.
  
  `modeldir` is the directory in which to look for the model file (and where we intend to
  save model checkpoints from the next epochs).

  - If `modeldir` directory points to a file, barf and die.

  - If `modeldir` directory is empty or missing, use function `makemodelfn`
    to create a new model instance.

  - If the `modeldir` directory is empty or absent, create the directory if needed and return
    a model newly created using `makemodelfn`.

  - If the dir `modeldir` contains files named epoch_<ID>.bson, find the epoch with the greatest id and load that.
  """

  pathtype = gspathtype(modeldir)

  if pathtype in [PATH_FILE, PATH_OTHER]
    throw(ArgumentError("Model dir specified is not a directory: $modeldir"))
  elseif pathtype in [PATH_NONEXISTENT, PATH_DIR_EMPTY]
    pathtype == PATH_NONEXISTENT && gsmkpath(modeldir)
    return makemodelfn()
  end

  checkpoints = [x for x in readdir(modeldir) if startswith(x, prefix) && endswith(x, ext)]
  ids = [parse(Int, x[1+length(prefix): length(x)-length(ext)]) for x in checkpoints]

  fname = checkpoints[findmax(ids)[2]]
  fname = joinpath(modeldir, fname)
  
  loadmodel(fname)  
end

function train(args::TrainArgs, am::Models.AnnotatedModel; kwargs...)
  @info "Number of threads: $(Threads.nthreads())"
  #tracker = StatsTracker()
  tracker = nothing

  @info "Setting up data"
  trainset, testset, trainset_subsample = prepare_data(args, tracker)
  
  Models.setmeta!(am, :train_args, args)
  am.model = gpu(am.model)

  #  model_dir = joinpath(args.modeldir, Dates.format(Dates.now(), "yyyymmdd-HHMM"))

  logdir=joinpath(args.modeldir, "tb")
  if isgcs(args.modeldir)
    logdir = mktempdir()
  end
  
  logger = TeeLogger(current_logger(),
                     TBLogger(logdir, tb_append))

  opt = ADAM(args.lr)
  @info("Starting training...")
  f1_old = 0

  first_iteration = true
  
  for i = (am.epoch+1):args.epochs
    am.epoch = i
    p = Progress(length(trainset), dt = 1.0, desc = "Epoch $i: ")

    lasttime = Ref(time_ns())

    function iteration_callback()
      if first_iteration
        @info @sprintf("Training started in %0.2f seconds", (time_ns() - lasttime[])/1e9)
        first_iteration = false
      end
      curtime = time_ns()
      elapsed = (curtime - lasttime[]) / 1e9
      report!(tracker, "train_iteration_time", elapsed)
      next!(p)
      lasttime[] = time_ns()
    end
    
    Flux.train!(bce_loss(am.model), params(am.model), trainset, opt, cb = iteration_callback)
    
    with_logger(logger) do
      p, r, f1, lossval = prf1(am.model, trainset_subsample, lossfn=Flux.Losses.binarycrossentropy)
      @info "train" precision=p recall=r f1=f1 loss=lossval
      p, r, f1, lossval = prf1(am.model, testset, lossfn=Flux.Losses.binarycrossentropy)
      @info "test" precision=p recall=r f1=f1 loss=lossval

      if f1 > f1_old || !args.only_save_model_if_better
        Models.savemodel(am, args.modeldir)
      end
      f1_old = max(f1, f1_old)
    end

    if isgcs(args.modeldir)
      if length(readdir(logdir)) > 0
        gcscopy("$(logdir)/*", "$(args.modeldir)/tb/")
      else
        @warn "Nothing found in tensorboard dir $(logdir)"
      end
    end

    trainset, testset, trainset_subsample = reset(trainset), reset(testset), reset(trainset_subsample)
  end

  return args.modeldir
end
