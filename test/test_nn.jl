using Test
using Hair, Hair.TestUtil
using Images, Flux
using Parameters: @with_kw

H = Hair



@with_kw struct PlusLayer
  w::Float32 = 1.0
end
Flux.@functor PlusLayer
(s::PlusLayer)(x::AbstractArray) = x .+ s.w


@testset "NN funcs" begin





  let x = ones(Float32, 3, 3, 2, 1)
    m = H.SkipUnit(Chain(PlusLayer(), PlusLayer()), Chain(PlusLayer(5), PlusLayer(5)))

    expected = ones(Float32, 3, 3, 4, 1)
    expected[:, :, 1:2, :] .*= 3
    expected[:, :, 3:4, :] .*= 13

    @test m(x) == expected
  end

end
