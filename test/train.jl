using Test
using Hair, Hair.TestUtil, Hair.Models
using Images
using Flux
import Zygote


H = Hair


@testset "Prepare data" begin

  tdir = H.TestUtil.write_dummy_images_masks(100, 50)

  trainset, testset =
    H.prepare_data(H.TrainArgs(img_dir = tdir, test_set_ratio = 0.4, batch_size = 4))

  @test length(trainset) == 15
  @test length(testset) == 10

  X_train, Y_train = trainset
  X_test, Y_test = testset

  @test size(X_train[1]) == (50, 50, 3, 4)
  @test size(Y_train[2]) == (50, 50, 1, 4)
  @test size(X_test[1]) == (50, 50, 3, 4)
  @test size(Y_test[2]) == (50, 50, 1, 4)
  @test eltype(X_train[1]) == Float32
  @test eltype(X_train[2]) == Float32
end


@testset "maybeloadmodel" begin

  test_makemodelfn() = Hair.AnnotatedModel("I am a dummy model", Hair.Models.SimpleArgs())

  # tdir can't be a file
  let tdir = mktempdir()
    fname = joinpath(tdir,"blah")
    open(fname, "w") do io
      write(io, "A file with something in it")
    end
    @test_throws ArgumentError Hair.maybeloadmodel(test_makemodelfn, fname)
  end

  # tdir is empty
  let tdir = mktempdir()
    am = Hair.maybeloadmodel(test_makemodelfn, tdir)
    @test am.model == "I am a dummy model"
  end

  # tdir is empty
  let tdir = mktempdir()
    modeldir = joinpath(tdir, "abracadabra")
    am = Hair.maybeloadmodel(test_makemodelfn, modeldir)
    @test am.model == "I am a dummy model"
    @test isdir(modeldir)
    @test readdir(modeldir) |> length == 0
  end

  dummy_model(epochid::Int) = AnnotatedModel(model=Chain(MaxPool((2,2))), metadata=Dict(:model_args=>"Model epoch $(epochid)"), epoch=epochid)
  # load the latest saved model, only one exists
  let tdir = mktempdir()
    Hair.Models.savemodel(dummy_model(1), tdir)
    @test Hair.maybeloadmodel(test_makemodelfn, tdir).metadata[:model_args] == "Model epoch 1"
    Hair.Models.savemodel(dummy_model(2), tdir)
    Hair.Models.savemodel(dummy_model(3), tdir)
    @test Hair.maybeloadmodel(test_makemodelfn, tdir).metadata[:model_args] == "Model epoch 3"
    Hair.Models.savemodel(dummy_model(9999), tdir)
    Hair.Models.savemodel(dummy_model(1009990), tdir)
    @test Hair.maybeloadmodel(test_makemodelfn, tdir).metadata[:model_args] == "Model epoch 1009990"
  end  
end

