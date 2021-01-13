using Test
using Hair, Hair.TestUtil
using Images

H=Hair


@testset "Prepare data" begin

  tdir = H.TestUtil.write_dummy_images_masks(100, 50)
  
  trainset, testset = H.prepare_data(H.TrainArgs(img_dir=tdir, test_set_ratio=0.4, batch_size=4))

  @test length(trainset) == 15
  @test length(testset) == 10

  X_train, Y_train = trainset
  X_test, Y_test = testset

  @test size(X_train[1]) == (50,50,3,4)
  @test size(Y_train[2]) == (50,50,1,4)
  @test size(X_test[1]) == (50,50,3,4)
  @test size(Y_test[2]) == (50,50,1,4)
  @test eltype(X_train[1]) == Float32
  @test eltype(X_train[2]) == Float32
end
