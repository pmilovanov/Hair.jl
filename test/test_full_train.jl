using Test
using Hair, Hair.TestUtil
using Images, Flux
using BSON

using Traceur
#using Profile

H = Hair
M = Hair.Models

@testset "A couple of epochs trained on a basic model" begin

  tdir = H.TestUtil.write_dummy_images_masks2(500, 256)

  @info "Images dir: $(tdir)"

  savepath = mktempdir(cleanup = false)

  #model = H.Models.build_model_simple([2, 2, 2, 2])
  model = M.simple(M.SimpleArgs())

  #  Profile.init(n=20000000)

  model_dir = H.train(
    H.TrainArgs(
      img_dir = tdir,
      test_set_ratio = 0.5,
      epochs = 2,
      savepath = savepath,
      only_save_model_if_better = false,
    ),
    model,
  )

  @info "Model dir: $(model_dir)"

  for p in ["epoch_001.bson", "epoch_002.bson"]
    fname = joinpath(model_dir, p)
    @test isfile(fname)
    @test stat(fname).size > 1000
  end

  BSON.@load joinpath(model_dir, "epoch_002.bson") model
  model = model |> gpu

  dataset = H.ImageAndMaskLoader(H.readdir_nomasks(tdir), batchsize = 4, bufsize = 10)
  for (x, y) in dataset
    ŷ = model(gpu(x))
    @test size(ŷ) == (256, 256, 1, 4)
    break
  end


  #  Profile.print()
end
