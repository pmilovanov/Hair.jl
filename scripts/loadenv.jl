using Hair; H = Hair

using Flux, NNlib
using BSON: @load, @save
using Images, ImageView, MosaicViews

show_infer_overlay(gmodel, imgpath, size=2048; device=gpu) = imshow(H.infer_overlay(gmodel, imgpath, size, device=device)[2])


function show_infer_overlay3(gmodel, imgpath; device=gpu, nrow=1)
  imgs = [H.infer_overlay(gmodel, imgpath, side, device=device)[2] for side in 1024 .* [1,2,3]]
  imshow(mosaicview(imgs, nrow=nrow))
end
