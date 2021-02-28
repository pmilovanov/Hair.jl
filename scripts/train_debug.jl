using Hair
using NNlib
using Flux
using ArgParse
using Hair.Models: AnnotatedModel, ModelArgs
using Profile
using ProgressMeter

function flags()
  s = ArgParseSettings()
  @add_arg_table s begin
    "--data"
    arg_type=String
#    required=true
    default=expanduser("~/data/hair/hairy/exp/full256_0121")

    "--modelsavedir"
    arg_type=String
 #   required=true
    default=expanduser("~/data/hair/models/memdebug/")    
  end
  parse_args(ARGS, s)
end

struct DumbModelArgs <: ModelArgs end
dumbmodel() = AnnotatedModel(Conv((3,3), 3=>1, σ; pad=(1,1)), DumbModelArgs())

if abspath(PROGRAM_FILE) == @__FILE__

  args = flags()

  model = dumbmodel() |> gpu

  Profile.clear_malloc_data()

  EPOCHS=10

  model = Chain(Conv((3,3), 3=>1, σ; pad=(1,1)),
                           GlobalMeanPool()) 


  trainargs = 
    Hair.TrainArgs(
      test_set_ratio = 0.97,
      img_dir = expanduser(args["data"]),
      savepath = expanduser(args["modelsavedir"]),
      batch_size = 16,
      epochs = 1000,
    )

  trainset, testset = Hair.prepare_data(trainargs, nothing)

  # trainset = trainset.inner

  for i=1:EPOCHS
    total = 0
    @info "Epoch $i"

    p = Progress(length(trainset), dt=0.5)
    j = 0
    for (X, Y) in trainset
      #yh = model(X)
      j += 1
      total += 1
      next!(p)
    end

    @info "Total $(cpu(total))"
    GC.gc()
    global trainset = Hair.reset(trainset)
  end
  
end
