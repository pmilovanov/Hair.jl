using Test
using Hair, Hair.TestUtil
using Images, Flux
using BSON

#using Traceur
#using Profile

H = Hair
M = Hair.Models


function assert_checkpoints_exist(checkpoint_fnames, model_dir)
  for p in checkpoint_fnames
    fname = joinpath(model_dir, p)
    @test isfile(fname)
    @test stat(fname).size > 1000
  end
end

@testset "A couple of epochs trained on a basic model" begin

  tdir = H.TestUtil.write_dummy_images_masks2(32, 128)

  @info "Images dir: $(tdir)"

  savepath = mktempdir(cleanup = false)

  #model = H.Models.build_model_simple([2, 2, 2, 2])
  modelfn() = M.simple(M.SimpleArgs())
  model = H.maybeloadmodel(modelfn, savepath)
  
  #  Profile.init(n=20000000)

  train_args = H.TrainArgs(
      img_dir = tdir,
      test_set_ratio = 0.5,
      epochs = 2,
      modeldir = savepath,
      only_save_model_if_better = false,
      batch_size = 16
   )
  
  model_dir = H.train(
    train_args,
    model
  )

  @test model.epoch == 2
  
  @info "Model dir: $(model_dir)"

  assert_checkpoints_exist(["epoch_001.bson", "epoch_002.bson"], model_dir)

  BSON.@load joinpath(model_dir, "epoch_002.bson") model
  @test model.epoch == 2
  model = model.model |> gpu

  dataset = H.SegmentationDataLoader(H.readdir_nomasks(tdir), batchsize = 4) |> H.GPUDataLoader
  for (x, y) in dataset
    ŷ = model(gpu(x))
    @test size(ŷ) == (128, 128, 1, 4)
    break
  end

  

  # Try training when modeldir points to a dir with existing checkpoints
  model2 = H.maybeloadmodel(modelfn, savepath)
  @test model2.epoch == 2
  model_dir2 = H.train(H.TrainArgs(
      img_dir = tdir,
      test_set_ratio = 0.5,
      epochs = 3,
      modeldir = savepath,
      only_save_model_if_better = false,
      batch_size = 8
  ), model2)
  @test model_dir2 == model_dir
  assert_checkpoints_exist(["epoch_001.bson", "epoch_002.bson", "epoch_003.bson"], model_dir)
end


