using Images, ImageView
using Hair

using Test
using LinearAlgebra

const GA = GrayA
@testset "component image ops" begin
    img = Gray.(reshape(Vector(1:100), (10, 10)) ./ 100)
    component_mask = Matrix(1.0I, 3, 3)  # 3x3 identity mtx

    component = Hair.Component(1, 3, [(3,3), (5,5)], component_mask)
    
    compimg = [GA(0.23, 1.0)  GA(0.33, 0.0)  GA(0.43, 0.0)
               GA(0.24, 0.0)  GA(0.34, 1.0)  GA(0.44, 0.0)
               GA(0.25, 0.0)  GA(0.35, 0.0)  GA(0.45, 1.0)]

    @test Hair.image(img, component) == compimg
end
