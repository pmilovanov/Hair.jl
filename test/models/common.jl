using Test
using Hair, Hair.TestUtil, Hair.Models
using Mocking, Flux

H = Hair

Mocking.activate()

@testset "Save model" begin

  function dummy()
    m = H.Models.simple()
    m.epoch=8
    m.model = gpu(m.model)
    m
  end

  # save model, happy path -- local
  let tdir = mktempdir()
    m = dummy()
    H.Models.savemodel(m, tdir)
    @test isfile(joinpath(tdir, "epoch_008.bson"))
    @test isfile(joinpath(tdir, "epoch_008.json"))
  end

  # file to save to already exists -- local
  let tdir = mktempdir()
    m = dummy()
    open(joinpath(tdir, "epoch_008.bson")) do io
      write("boo")
    end    
    @test_throws InvalidStateException H.Models.savemodel(m, tdir)
  end

  # save model, happy path -- gcs
  let tdir = mktempdir()
    m = dummy()
    cmds = []
    p_run = @patch run(cmd::Cmd) = push!(cmds, cmd)
    p_open = @patch function open(f::Function, c::Base.AbstractCmd, args...; kwargs...)
      push!(cmds, c)
      io = IOBuffer()
      f(io)
    end

    apply([p_open, p_run]) do
      H.Models.savemodel(m, "gs://hairy/test")

      @test cmds[1] == `gsutil ls gs://hairy/test/epoch_005.bson`
      
      @test cmds[2].exec[1:3] == ["gsutil", "-m", "cp"]
      @test endswith(cmds[2].exec[4], "epoch_005.bson")
      @test cmds[2].exec[5] == "gs://hairy/test/"

      @test cmds[3].exec[1:3] == ["gsutil", "-m", "cp"]
      @test endswith(cmds[3].exec[4], "epoch_005.json")
      @test cmds[3].exec[5] == "gs://hairy/test/"
    end
  end

  # file to save to already exists -- gcs
  let tdir = mktempdir()
    m = dummy()
    cmds = []
    p_open = @patch function open(f::Function, c::Base.AbstractCmd, args...; kwargs...)
      push!(cmds, c)
      io = IOBuffer("gs://hairy/test/epoch_005.bson\n")
      f(io)
    end
    apply(p_open) do
      @test_throws InvalidStateException H.Models.savemodel(m, "gs://hairy/test")
    end
  end

end
