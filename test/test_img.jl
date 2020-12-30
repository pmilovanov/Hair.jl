using Images
using Hair
using Hair: bbox

using Test
using LinearAlgebra


const GA = GrayA

H = Hair

@testset "Image manipulation" begin

@test H.torange(bbox(5, 7, 9, 11)) == [5:9, 7:11]

@testset "Component ops" begin
    img = Gray.(reshape(Vector(1:100), (10, 10)) ./ 100)
    component_mask = Matrix(1.0I, 3, 3)  # 3x3 identity mtx

    component = H.Component(1, 3, [(3, 3), (5, 5)], component_mask)

    compimg = [
        GA(0.23, 1.0) GA(0.33, 0.0) GA(0.43, 0.0)
        GA(0.24, 0.0) GA(0.34, 1.0) GA(0.44, 0.0)
        GA(0.25, 0.0) GA(0.35, 0.0) GA(0.45, 1.0)
    ]

    @test Hair.image(img, component) == compimg

    # imgslice
    @test H.imgslice(
        [
            0 0 0 0 0
            0 1 1 1 0
            0 1 0 1 0
            0 1 1 1 0
            0 0 0 0 0
        ],
        [(3, 2), (4, 5)],
    ) == [
        1 0 1 0
        1 1 1 0
    ]
end

@testset "Place one image on another" begin
    @testset "Helper functions" begin
        zeroboxes = (bbox(1, 1, 0, 0), bbox(1, 1, 0, 0))

        @test H.interval_overlap((1, 10), (1, 10)) == ((1, 10), (1, 10))
        @test H.interval_overlap((1, 10), (10, 20)) == ((10, 10), (1, 1))
        @test H.interval_overlap((10, 20), (1, 10)) == ((1, 1), (10, 10))
        @test H.interval_overlap((21, 30), (1, 40)) == ((1, 10), (21, 30))
        @test H.interval_overlap((0, 41), (1, 40)) == ((2, 41), (1, 40))
        @test H.interval_overlap((-99, 40), (1, 40)) == ((101, 140), (1, 40))
        @test H.interval_overlap((1, 40), (1, 400)) == ((1, 40), (1, 40))
        @test H.interval_overlap((-4, 5), (1, 10)) == ((6, 10), (1, 5))
        @test H.interval_overlap((6, 15), (1, 10)) == ((1, 5), (6, 10))

        # Boxes are the same
        @test H.box_overlap(bbox(1, 1, 1000, 1000), bbox(1, 1, 1000, 1000)) ==
            (bbox(1, 1, 1000, 1000), bbox(1, 1, 1000, 1000))
        @test H.box_overlap((1000, 1000), (1000, 1000), (1, 1)) ==
            (bbox(1, 1, 1000, 1000), bbox(1, 1, 1000, 1000))

        # Src completely inside dest
        @test H.box_overlap(bbox(201, 201, 400, 400), bbox(1, 1, 1000, 1000)) ==
            (bbox(1, 1, 200, 200), bbox(201, 201, 400, 400))
        @test H.box_overlap((200, 200), (1000, 1000), (201, 201)) ==
            (bbox(1, 1, 200, 200), bbox(201, 201, 400, 400))

        # No overlap
        @test H.box_overlap((300, 200), (1000, 1000), (-500, 2)) == zeroboxes
        @test H.box_overlap((300, 200), (1000, 1000), (-500, -500)) == zeroboxes
        @test H.box_overlap((300, 200), (1000, 1000), (1001, 2001)) == zeroboxes
        @test H.box_overlap((300, 200), (1000, 1000), (1001, 300)) == zeroboxes

        # 1 pixel overlap
        @test H.box_overlap((100, 100), (1000, 1000), (1000, 1000)) ==
            (bbox(1, 1, 1, 1), bbox(1000, 1000, 1000, 1000))
        # 1 line overlap
        @test H.box_overlap((100, 100), (1000, 1000), (501, 1000)) ==
            (bbox(1, 1, 100, 1), bbox(501, 1000, 600, 1000))
        # Partial overlap
        @test H.box_overlap((100, 100), (1000, 1000), (951, 951)) ==
            (bbox(1, 1, 50, 50), bbox(951, 951, 1000, 1000))
        @test H.box_overlap((100, 100), (1000, 1000), (-49, -39)) ==
            (bbox(51, 41, 100, 100), bbox(1, 1, 50, 60))
    end

    # Place tests
    @testset "Place" begin

        destconst() = [0.0  0.0  0.0  0.0  0.0
                       0.0  0.0  0.0  0.0  0.0
                       0.0  0.0  0.5  0.5  0.5
                       0.0  0.0  0.5  0.5  0.5
                       0.0  0.0  0.5  0.5  0.5]
        
        dest = Gray.(destconst())
        cross = Float64.([1  0  1
                          0  1  0
                          1  0  1])

        src_cross_no_bg = GrayA.(cross, cross)
        src_opaque = GrayA.(cross, ones(Float64, 3, 3))
        src_fullytransparent = GrayA.(cross, zeros(Float64, 3, 3))
        src_semitransparent = GrayA.(cross, fill(0.5, 3, 3))

        @test Float64.(H.place(src_cross_no_bg, dest, (1, 1))) == [1.0  0.0  1.0  0.0  0.0
                                                                   0.0  1.0  0.0  0.0  0.0
                                                                   1.0  0.0  1.0  0.5  0.5
                                                                   0.0  0.0  0.5  0.5  0.5
                                                                   0.0  0.0  0.5  0.5  0.5]
        # Verify dest is still unchanged
        @test dest ==  Gray.([0.0  0.0  0.0  0.0  0.0
                              0.0  0.0  0.0  0.0  0.0
                              0.0  0.0  0.5  0.5  0.5
                              0.0  0.0  0.5  0.5  0.5
                              0.0  0.0  0.5  0.5  0.5])

        # Put a transparent cross on dest => nothing changes
        @test Float64.(H.place(src_fullytransparent, dest, (1, 1))) ==  Gray.([0.0  0.0  0.0  0.0  0.0
                                                                               0.0  0.0  0.0  0.0  0.0
                                                                               0.0  0.0  0.5  0.5  0.5
                                                                               0.0  0.0  0.5  0.5  0.5
                                                                               0.0  0.0  0.5  0.5  0.5])

        # Fully opaque image
        @test Float64.(H.place(src_opaque, dest, (2, 2))) == [0.0  0.0  0.0  0.0  0.0
                                                              0.0  1.0  0.0  1.0  0.0
                                                              0.0  0.0  1.0  0.0  0.5
                                                              0.0  1.0  0.0  1.0  0.5
                                                              0.0  0.0  0.5  0.5  0.5]

        # Semitransparent
        @test Float64.(H.place(src_semitransparent, dest, (2, 2))) == [0.0  0.0  0.0   0.0   0.0
                                                                       0.0  0.5  0.0   0.5   0.0
                                                                       0.0  0.0  0.75  0.25  0.5
                                                                       0.0  0.5  0.25  0.75  0.5
                                                                       0.0  0.0  0.5   0.5   0.5]

        # Semitransparent, partial overlap
        @test Float64.(H.place(src_semitransparent, dest, (2, 5))) == [0.0  0.0  0.0  0.0  0.0
                                                                       0.0  0.0  0.0  0.0  0.5
                                                                       0.0  0.0  0.5  0.5  0.25
                                                                       0.0  0.0  0.5  0.5  0.75
                                                                       0.0  0.0  0.5  0.5  0.5]


        # Place src outside of dest
        @test Float64.(H.place(src_cross_no_bg, dest, (8, 8))) == destconst()
        @test Float64.(H.place(src_cross_no_bg, dest, (-3, -3))) == destconst()
        @test Float64.(H.place(src_cross_no_bg, dest, (2, 6))) == destconst()
        
        
    end
end

    # @testset "Misc" begin

    #     cross = Float64.([1  0  1
    #                       0  1  0
    #                       1  0  1])

    #     imghsl = HSL.(rand(Float64,3,3), rand(Float64,3,3), cross)
    #     imgrgb = convert.(RGB, imghsl)


    #     @test H.matte_from_luminance(RGBA.(imgrgb, ones(Float64, 3,3))) == RGBA.(imgrgb, cross)
    #     @test H.matte_from_luminance(RGBA.(imgrgb, fill(0.5, 3,3))) == RGBA.(imgrgb, [0.5 0.0 0.5
    #                                                                                   0.0 0.5 0.0
    #                                                                                   0.5 0.0 0.5])
    # end

end
