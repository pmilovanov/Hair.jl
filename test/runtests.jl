using Test

@testset "All" begin
  include("test.jl")
  include("test_gendata.jl")
  include("test_file.jl")
  include("test_img.jl")
  include("test_train.jl")
  include("test_dataloaders.jl")
  include("test_nn.jl")
end
