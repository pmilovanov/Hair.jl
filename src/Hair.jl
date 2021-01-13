module Hair

include("util.jl")
include("synthetic.jl")
include("img.jl")
include("gendata.jl")
include("file.jl")
include("train.jl")
include("test_util.jl")

export sample_image, GridStrategy

end
