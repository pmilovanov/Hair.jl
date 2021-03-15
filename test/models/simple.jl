using Test
using Hair, Hair.TestUtil, Hair.Models
using Mocking, Flux

H = Hair

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
