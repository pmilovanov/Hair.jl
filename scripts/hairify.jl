
include("loadenv.jl")

@load H.latestmodel(dir="~/data/hair/models/5555") model

function optfn(x)
  z = gmodel(x)
  -1 * sum(z .* z)
end

opt = ADAM(3e-3)

lr = 0.01f0

img = testimage("lena")
X = H.imgtoarray(imresize(img, (512,512))) |> gpu

for i = 1:300
  g = gradient(optfn, X)
  Flux.update!(opt, X, g[1])
end
