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
