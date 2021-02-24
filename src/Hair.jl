module Hair

include("gcs.jl")
include("util.jl")
include("channels.jl")

include("img.jl")


include("file.jl")
include("gendata.jl")


include("dataloaders.jl")
include("nn.jl")
include("synthetic.jl")
include("test_util.jl")

include("models/Models.jl")

include("train.jl")
include("eval.jl")

export GridStrategy

end
