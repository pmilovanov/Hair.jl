using Images, ImageView
using Hair
using Hair: bbox

using Test
using LinearAlgebra


const GA = GrayA

H = Hair

@testset "component image ops" begin
    img = Gray.(reshape(Vector(1:100), (10, 10)) ./ 100)
    component_mask = Matrix(1.0I, 3, 3)  # 3x3 identity mtx

    component = H.Component(1, 3, [(3, 3), (5, 5)], component_mask)

    compimg = [
        GA(0.23, 1.0) GA(0.33, 0.0) GA(0.43, 0.0)
        GA(0.24, 0.0) GA(0.34, 1.0) GA(0.44, 0.0)
        GA(0.25, 0.0) GA(0.35, 0.0) GA(0.45, 1.0)
    ]

    @test Hair.image(img, component) == compimg
end


@testset "imgslice" begin
    img = [
        0 0 0 0 0
        0 1 1 1 0
        0 1 0 1 0
        0 1 1 1 0
        0 0 0 0 0
    ]

    @test H.imgslice(img, [(3, 2), (4, 5)]) == [
        1 0 1 0
        1 1 1 0
    ]
end


zeroboxes = (bbox(0, 0, 0, 0), bbox(0, 0, 0, 0))

@testset "place" begin
    @test Hair.interval_overlap((1, 10), (1, 10)) == ((1, 10), (1, 10))
    
    # @test_throws AssertionError H.srcdestboxes((-1, 5), (5, 5), (1, 1))
    # @test_throws AssertionError H.srcdestboxes((5, 0), (1, 1), (0, 0))
    # @test_throws AssertionError H.srcdestboxes((5, 5), (-5, 1), (5, 5))
    # @test_throws AssertionError H.srcdestboxes((9, 5), (2, 0), (5, 5))
    # @test_throws AssertionError H.srcdestboxes((-1, -1), (-1, -1), (5, 5))
    # @test_throws AssertionError H.srcdestboxes((0, 0), (0, 0), (0, 0))
    
    # @test H.srcdestboxes((200, 100), (2000, 1000), (-300, -300)) ==
    #     (bbox(200, 100, 199, 99), bbox(1, 1, 0, 0))
    # @test H.srcdestboxes((200, 100), (2000, 1000), (-200, -100)) ==
    #     (bbox(200, 100, 199, 99), bbox(1, 1, 0, 0))
    # @test H.srcdestboxes((200, 100), (2000, 1000), (3000, 3000)) ==
    #     (bbox(1, 1, 0, 0), bbox(2000, 1000, 1999, 999))

    # @test H.srcdestboxes((1, 1), (2000, 1000), (5, 5)) ==
    #     (bbox(1,1,1,1), bbox(5,5,5,5))
    

    
    # @test H.srcdestboxes((200, 100), (2000, 1000), (-199, -99)) |> size == (0,0)
    # @test H.srcdestboxes((200, 100), (2000, 1000), (2001, 1001)) |> size == (0,0)
    # @test H.srcdestboxes((200, 100), (2000, 1000), (3000, 3000)) |> size == (0,0)
end
