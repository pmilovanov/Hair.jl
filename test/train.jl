using Test
using Hair, Hair.TestUtil, Hair.Models
using Images
using Flux
import Zygote
using SimpleMock

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

  @test repr(H.Models.conv_block(3, (3, 3), 128 => 256, relu, pad = (1, 1), stride = (1, 1))) ==
        repr(
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
  let x = rand(Float32, 128, 128, 3, 1), y = rand(Float32, 128, 128, 1, 1)
    m = H.Models.simple(
      H.Models.SimpleArgs(
        blocksizes = [3, 3, 3, 3, 3],
        kernelsizes = [(5, 5), (5, 5), (5, 5), (5, 5), (5, 5)],
      ),
    )
    ŷ = m(x)

    checkok(x) = (!all(x .== 0) && all(x .!= NaN))

    @test checkok(ŷ)

    opt = ADAM(3e-3)
    ps = Zygote.Params(params(m))
    loss = H.bce_loss(m)
    gs = gradient(ps) do
      loss(x, y)
    end

    for p in ps
      @test checkok(gs[p])
    end

    Flux.update!(opt, ps, gs)

    for p in ps
      @test checkok(p)
    end
  end
end
