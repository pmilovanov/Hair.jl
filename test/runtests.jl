using Test

#@testset "All" begin
include("test.jl")
include("gendata.jl")
include("file.jl")
include("img.jl")
include("train.jl")
include("dataloaders.jl")
include("nn.jl")
include("gcs.jl")

include("models/runtests.jl")
#end
