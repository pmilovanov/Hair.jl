using Test
using Hair, Hair.TestUtil
using Images, Flux

H = Hair

@testset "A couple of epochs trained on a basic model" begin

  tdir = H.TestUtil.write_dummy_images_masks(200, 128)
  savepath=joinpath(tdir,"models")

  H.train(H.TrainArgs(img_dir=tdir, test_set_ratio=0.5, epochs=2))

  for p in ["epoch_001.bson", "epoch_002.bson"]
    fname = joinpath(savepath, p)
    @test isfile(p)
    @test stat(p).size > 1000
  end
end
