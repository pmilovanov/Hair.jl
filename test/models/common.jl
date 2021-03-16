using Test
using Hair, Hair.TestUtil, Hair.Models
using Mocking, Flux

H = Hair

Mocking.activate()

@testset "Save model" begin

  let tdir = mktempdir()
    m = H.Models.simple()
    m.model = gpu(m.model)

    H.Models.savemodel(m, tdir, 8)
    @test isfile(joinpath(tdir, "epoch_008.bson"))
    @test isfile(joinpath(tdir, "epoch_008.json"))
  end

  let tdir = mktempdir()
    m = H.Models.simple()
    m.model = gpu(m.model)

    cmds = []
    p = @patch function run(cmd::Cmd)
      push!(cmds, cmd)
      run(`echo`)
    end

    apply(p) do
      H.Models.savemodel(m, "gs://hairy/test", 5)

      @test cmds[1].exec[1:3] == ["gsutil", "-m", "cp"]
      @test endswith(cmds[1].exec[4], "epoch_005.bson")
      @test cmds[1].exec[5] == "gs://hairy/test/"

      @test cmds[2].exec[1:3] == ["gsutil", "-m", "cp"]
      @test endswith(cmds[2].exec[4], "epoch_005.json")
      @test cmds[2].exec[5] == "gs://hairy/test/"
    end
  end

end


@testset "Stack channels" begin
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

