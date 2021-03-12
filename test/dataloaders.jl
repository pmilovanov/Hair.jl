using Test
using Hair
using Flux.Data: DataLoader
using MLDataPattern
H = Hair

@testset "Dataloaders" begin

  @testset "GPUDataLoader" begin

    X = rand(Float32, 256, 256, 3, 1000)
    Y = rand(Float32, 256, 256, 1, 1000)
    gdl = H.GPUDataLoader(DataLoader((X, Y), batchsize = 64, shuffle = true, partial = false))

    @test length(gdl) == 15
    for data in gdl
      @test size(data[1]) == (256, 256, 3, 64)
      break
    end
  end


  @testset "Segmentation DataLoaders 1" for LoaderType in [
    H.AsyncSegmentationDataLoader,
    H.SegmentationDataLoader,
  ]
    dirname = H.TestUtil.write_dummy_images_masks(16, 128)

    filenames = [f for f in readdir(dirname, join = true) if !contains(f, "-mask")]
    loader = LoaderType(filenames, batchsize = 5, shuffle = true)

    n = 0
    for (x, y) in loader
      n += 1
      @test size(x) == (128, 128, 3, 5)
      @test size(y) == (128, 128, 1, 5)
      @test eltype(x) == Float32
      @test eltype(y) == Float32
      @test x != zeros(Float32, 128, 128, 3, 5)
      @test y != zeros(Float32, 128, 128, 1, 5)
    end
    @test n == 3
  end

end;
