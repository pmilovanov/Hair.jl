using Test
using Hair, Hair.TestUtil
using Images, Flux

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


@testset "ML funcs" begin

  @test repr(H.conv_block(3, (3, 3), 128 => 256, relu, pad = (1, 1), stride = (1, 1))) == repr(
    Chain(
      Conv((3, 3), 128 => 256, relu, pad = (1, 1), stride = (1, 1)),
      BatchNorm(256),
      Conv((3, 3), 256 => 256, relu, pad = (1, 1), stride = (1, 1)),
      BatchNorm(256),
      Conv((3, 3), 256 => 256, relu, pad = (1, 1), stride = (1, 1)),
      BatchNorm(256),
    ),
  )

  z = rand(Float32, (256, 256, 3, 1))
  stack_layer = H.StackChannels(
    Conv((3, 3), 3 => 10, relu, pad = SamePad()),
    Conv((3, 3), 3 => 12, relu, pad = SamePad()),
  )
  @test size(stack_layer(z)) == (256, 256, 22, 1)



end


@testset "ML funcs CUDA" begin
  # begin
  #   z = rand(Float32, (512, 512, 3, 1)) |> gpu
  #   m = H.build_model() |> gpu
  #   z2 = m(z) |> cpu
  #   @test !all(z2 .== 0)
  # end

  begin
    z = rand(Float32, (512, 512, 3, 1))
    m = H.build_model_simple()
    z2 = m(z)
    @test !all(z2 .== 0)
  end

end
