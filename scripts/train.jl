using Hair
using NNlib
using ArgParse

function flags()
  s = ArgParseSettings()
  @add_arg_table s begin
    "--data"
    arg_type = String
    required = true

    "--modelsavedir"
    arg_type = String
    required = true

    "--batch_size"
    arg_type = Int
    default = 16

    "--epochs"
    arg_type = Int
    default = 10

    "--previous_model"
    arg_type = String
    default = nothing
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

      savepath = expanduser(args["modelsavedir"]),
      #savepath = expanduser("~/data/hair/models/leakyrelu_55_77_256/"),
      previous_saved_model = args["previous_model"],
      batch_size = args["batch_size"],
      epochs = args["epochs"],
    ),
    model,
  )
end
