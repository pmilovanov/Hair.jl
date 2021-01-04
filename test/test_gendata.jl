using Images
using Hair
H = Hair

using Test

const GA = GrayA

H = Hair

testdata(datasize::Int) =
    reshape([i + j for i = 1:datasize for j = 1:datasize], (datasize, datasize))

@testset "Generating synthetic image training data" begin

    img = [
        2 3 4 5 6 7
        3 4 5 6 7 8
        4 5 6 7 8 9
        5 6 7 8 9 10
        6 7 8 9 10 11
        7 8 9 10 11 12
    ]


    @test H.sample_image(
        img,
        GridStrategy(n = 4, side = 3, overlap = 1, cover_edges = false),
    ) == [
        [
            2 3 4
            3 4 5
            4 5 6
        ],
        [
            4 5 6
            5 6 7
            6 7 8
        ],
        [
            4 5 6
            5 6 7
            6 7 8
        ],
        [
            6 7 8
            7 8 9
            8 9 10
        ],
        [
            5 6 7
            6 7 8
            7 8 9
        ],
        [
            7 8 9
            8 9 10
            9 10 11
        ],
        [
            5 6 7
            6 7 8
            7 8 9
        ],
        [
            7 8 9
            8 9 10
            9 10 11
        ],
        [
            8 9 10
            9 10 11
            10 11 12
        ],
    ]


    @test H.sample_image(img, GridStrategy(n = 4, side = 3, overlap = 1)) == [
        [
            2 3 4
            3 4 5
            4 5 6
        ],
        [
            4 5 6
            5 6 7
            6 7 8
        ],
        [
            4 5 6
            5 6 7
            6 7 8
        ],
        [
            6 7 8
            7 8 9
            8 9 10
        ],
    ]

end
