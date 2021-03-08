using Hair;
H = Hair;

using Flux, NNlib
using BSON: @load, @save
using Images


# @load "/home/pmilovanov/data/hair/models/fav/best.bson" model

# @load "/home/pmilovanov/data/hair/models/tmp/20210215-2131/epoch_1000.bson" model
#@load "/home/pmilovanov/data/hair/models/leakyrelu_55_77_256/20210218-1936/epoch_001.bson" model
@load "/home/pmilovanov/data/hair/models/leakyrelu_55555/20210307-0000/epoch_005.bson" model
gmodel = gpu(model.model)

gtdir = "/home/pmilovanov/data/hair/private/realgt"




fname="derain-port"
#fname="gauguin-3women"

rr = H.infer_compare_w_gt(
  gmodel,
  "$(gtdir)/$(fname).jpg",
  "$(gtdir)/$(fname)-mask.jpg"
)

save("~/tmp/hairout.jpg", rr.overlaid)

# imshow(rr.overlaid)


# rr = H.infer_compare_w_gt(
#   gmodel,
#   "$(gtdir)/Derain,_Andre_-_Port.jpg",
#   "$(gtdir)/Derain,_Andre_-_Port-mask.jpg",
# )

#imshow(rr.overlaid)

# metrics = H.eval_on_image_dir(gmodel, "/home/pmilovanov/data/hair/private/realgt")

# @show pairs(metrics)
