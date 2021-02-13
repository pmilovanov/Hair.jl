using Hair

if abspath(PROGRAM_FILE) == @__FILE__

  model = Hair.Models.selu_simple()

  Hair.train(
    Hair.TrainArgs(
      test_set_ratio = 0.05,
      #img_dir = expanduser("~/data/hair/hairy/exp/full128_0120"),
      img_dir = expanduser("~/data/hair/hairy/exp/full256_0121"),
      #img_dir = expanduser("~/data/hair/hairy/exp/10k_256_0211"),
      savepath = expanduser("~/data/hair/models/selu_simple"),
      #savepath = expanduser("~/data/hair/models/proper_stack_5555/"),
      #   previous_saved_model = "/home/pmilovanov/data/hair/models/5555/20210123-1314/epoch_004.bson"  ,
      batch_size = 32,
      epochs = 1000,
    ),
    model = model,
  )
end
