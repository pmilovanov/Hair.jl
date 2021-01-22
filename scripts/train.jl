using Hair

if abspath(PROGRAM_FILE) == @__FILE__
  Hair.train(Hair.TrainArgs(
    test_set_ratio = 0.05,
    img_dir = expanduser("~/data/hair/hairy/exp/full256_0121"),
    savepath = expanduser("~/data/hair/models/"),
    previous_saved_model = "/home/pmilovanov/data/hair/models/20210122-1051/epoch_005.bson",
    batch_size = 16,
  ))
end
