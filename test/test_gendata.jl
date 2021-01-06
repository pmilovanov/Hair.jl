using Images
using Hair
H = Hair

using Test

const GA = GrayA

H = Hair

testdata(datasize::Int) = reshape([i + j for i = 1:datasize for j = 1:datasize], (datasize, datasize))

@testset "Generating synthetic image training data" begin

  img = [
    2 3 4 5 6 7
    3 4 5 6 7 8
    4 5 6 7 8 9
    5 6 7 8 9 10
    6 7 8 9 10 11
    7 8 9 10 11 12
  ]

  img = [
    1 7 13 19 25 31
    2 8 14 20 26 32
    3 9 15 21 27 33
    4 10 16 22 28 34
    5 11 17 23 29 35
    6 12 18 24 30 36
  ]

  si(x, y) = img[x:x+2, y:y+2]
  @test si(3, 3) == [15 21 27; 16 22 28; 17 23 29]


  function compare_arr(a, b)
    for (ae, be) in zip(a, b)
      show(stdout, "text/plain", ae)
      print("\n")
      show(stdout, "text/plain", be)
      print("\n")
      println("---")
    end
  end

  @test H.sample_image(img, GridStrategy(side = 3, overlap = 1)) ==
        [si(1, 1), si(3, 1), si(1, 3), si(3, 3), si(4, 1), si(4, 3), si(1, 4), si(3, 4), si(4, 4)]

  @test H.sample_image(img, GridStrategy(side = 3, overlap = 1, cover_edges = false)) ==
        [si(1, 1), si(3, 1), si(1, 3), si(3, 3)]

  @test H.sample_image(img[1:5, :], GridStrategy(side = 3, overlap = 1)) ==
        [si(1, 1), si(3, 1), si(1, 3), si(3, 3), si(1, 4), si(3, 4)]


  @test H.sample_image(img[1:5, 1:5], GridStrategy(side = 3, overlap = 1)) == [si(1, 1), si(3, 1), si(1, 3), si(3, 3)]


end
