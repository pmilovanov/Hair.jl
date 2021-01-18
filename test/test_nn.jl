using Test
using Hair
using Flux.Data: DataLoader
H = Hair

@testset "NN funcs" begin


  X = rand(Float32, 256, 256, 3, 1000)
  Y = rand(Float32, 256, 256, 1, 1000)
  gdl = H.GPUDataLoader(DataLoader((X, Y), batchsize = 64, shuffle = true, partial = false))

  @test length(gdl) == 15
  for data in gdl
    @test size(data[1]) == (256, 256, 3, 64)
    break
  end


end
