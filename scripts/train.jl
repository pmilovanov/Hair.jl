using Hair
using NNlib
using ArgParse

function flags()
  s = ArgParseSettings()
  @add_arg_table s begin
    "--data"
    arg_type = String
    required = true

    "--modeldir"
    arg_type = String
    required = true
    help = """
           Directory to save the model to. If the directory already exists and contains
           epoch_XXX.bson files, trainer will load the latest epoch model and continue training from there.
           If --loadmodel is provided, --modeldir must be a dir path, and either not exist or be empty.
           """

    "--loadmodel"
    arg_type = String
    default = nothing
    help = "Path to a .bson model to load and continue training."

    
    "--batch_size"
    arg_type = Int
    default = 16

    "--epochs"
    arg_type = Int
    default = 10

  end
  parse_args(ARGS, s)
end


if abspath(PROGRAM_FILE) == @__FILE__

  args = flags()

  model = Hair.Models.simple(
    Hair.Models.SimpleArgs(
      blocksizes = [5, 5, 5, 5, 5],
      kernelsizes = [(5, 5), (5, 5), (5, 5), (5, 5), (5, 5)],
      σ = leakyrelu,
    ),
  )

  Hair.train(
    Hair.TrainArgs(
      test_set_ratio = 0.05,
      #img_dir = expanduser("~/data/hair/hairy/exp/full128_0120"),
      #img_dir = expanduser("~/data/hair/hairy/exp/full256_0121"),
      #img_dir = expanduser("/home/pmilovanov/data/hair/hairy/exp/1k_256"),

      img_dir = expanduser(args["data"]),

      #img_dir = expanduser("~/data/hair/hairy/exp/10k_256_0211"),
      #      savepath = expanduser("~/data/hair/models/selu_simple"),
      #avepath = expanduser("~/data/hair/models/tmp"),

      savepath = expanduser(args["modeldir"]),
      #savepath = expanduser("~/data/hair/models/leakyrelu_55_77_256/"),
      previous_saved_model = args["loadmodel"],
      batch_size = args["batch_size"],
      epochs = args["epochs"],
    ),
    model,
  )
end
