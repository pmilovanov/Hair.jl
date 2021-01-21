using Hair

if abspath(PROGRAM_FILE) == @__FILE__
  Hair.train(Hair.TrainArgs(test_set_ratio=0.05,
                       img_dir=expanduser("~/data/hair/hairy/exp/full128_0120"),
                       savepath=expanduser("~/data/hair/models/"),
                       batch_size=32)
)
end
