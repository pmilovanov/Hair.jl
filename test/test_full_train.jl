using Test
using Hair, Hair.TestUtil
using Images, Flux
using BSON

H = Hair

@testset "A couple of epochs trained on a basic model" begin

  tdir = H.TestUtil.write_dummy_images_masks(200, 128)

  savepath=mktempdir(cleanup=false)

  model_dir = H.train(H.TrainArgs(img_dir=tdir, test_set_ratio=0.5, epochs=2, savepath=savepath))

  for p in ["epoch_001.bson", "epoch_002.bson"]
    fname = joinpath(model_dir, p)
    @test isfile(fname)
    @test stat(fname).size > 1000
  end

  BSON.@load joinpath(model_dir, "epoch_002.bson") model
  model = model |> gpu

  dataset = H.ImageAndMaskLoader(H.readdir_nomasks(tdir), batchsize=4, bufsize=10)
  for (x,y) in dataset
    ŷ = model(gpu(x))
    @test size(ŷ) == (128, 128, 1, 4)
    break
  end
end
