module Hair

include("img.jl")
include("util.jl")

include("file.jl")
include("gendata.jl")

include("nn.jl")
include("synthetic.jl")
include("test_util.jl")
include("train.jl")


export sample_image, GridStrategy

end
