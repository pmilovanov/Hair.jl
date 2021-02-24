using Test
using Mocking

using Hair, Hair.TestUtil, Hair.Models
H = Hair
using Hair: GCSPath, @gs_str, cp, uri, gcscopy, untar


@testset "GCS functions" begin

  @test gs"bb/aa" == GCSPath("bb/aa")
  @test gs"gs://bb/aa" == GCSPath("bb/aa")
  @test gs"" == GCSPath("")
  @test gs"gs://" == GCSPath("")

  Mocking.activate()

  let tdir = mktempdir()
    patch = @patch run(c::Cmd) = run(`cp $(@__DIR__)/test.tar $(tdir)/`)
    apply(patch) do
      untar(gs"hairy/test.tar", tdir)
      @test sort(readdir(tdir)) == ["1","2","3"]
    end
  end 
end

