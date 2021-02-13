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


  let ŷ = Bool[
      0 0 0 0 0 0
      0 0 1 1 1 1
      0 0 1 1 1 1
      0 0 1 1 1 1
      0 0 1 1 1 1
    ],

    y = Bool[
      1 1 1 1 0 0
      1 1 1 1 0 0
      1 1 1 1 0 0
      0 0 0 0 0 0
      0 0 0 0 0 0
    ]

    tp = 4
    tn = 6
    fp = 12
    fn = 8

    ap_ŷ = 16
    ap_y = 12
    npixels = 30

    precision = 0.25
    recall = 1.0 / 3.0
    f1 = 2 * precision * recall / (precision + recall)

    @test H.binsegmetrics(ŷ, y) ==
          H.BinarySegmentationMetrics((precision, recall, f1, ap_ŷ, ap_y, tp, tn, fp, fn, npixels))
  end


  let N = 5,
    imgs = repeat(
      [(
        Gray{Float32}.([
          0 0 0 0 0 0
          0 0 1 1 1 1
          0 0 1 1 1 1
          0 0 1 1 1 1
          0 0 1 1 1 1
        ]),
        Gray{Float32}.([
          1 1 1 1 0 0
          1 1 1 1 0 0
          1 1 1 1 0 0
          0 0 0 0 0 0
          0 0 0 0 0 0
        ]),
      )],
      N,
    )

    tp = 4 * N
    tn = 6 * N
    fp = 12 * N
    fn = 8 * N

    ap_ŷ = 16 * N
    ap_y = 12 * N
    npixels = 30 * N

    precision = 0.25
    recall = 1.0 / 3.0
    f1 = 2 * precision * recall / (precision + recall)

    @test H.eval_on_images(imgs) ==
          H.BinarySegmentationMetrics((precision, recall, f1, ap_ŷ, ap_y, tp, tn, fp, fn, npixels))
  end

end
