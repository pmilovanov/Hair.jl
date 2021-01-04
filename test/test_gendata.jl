using Images
using Hair
H = Hair

using Test

const GA = GrayA

H = Hair

testdata(datasize::Int) =
    reshape([i + j for i = 1:datasize for j = 1:datasize], (datasize, datasize))

@testset "Generating synthetic image training data" begin

    img = testdata(6)

    @test [
        [2 3 4; 3 4 5; 4 5 6],
        [4 5 6; 5 6 7; 6 7 8],
        [4 5 6; 5 6 7; 6 7 8],
        [6 7 8; 7 8 9; 8 9 10],
    ] == H.sample_image(img, GridStrategy(4, 3, 1))

end
