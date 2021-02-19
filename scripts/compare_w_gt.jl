include("loadenv.jl")

# @load "/home/pmilovanov/data/hair/models/fav/best.bson" model

# @load "/home/pmilovanov/data/hair/models/tmp/20210215-2131/epoch_1000.bson" model
@load "/home/pmilovanov/data/hair/models/leakyrelu_55_77_256/20210218-1936/epoch_001.bson" model
gmodel = gpu(model)

gtdir = "/home/pmilovanov/data/hair/private/realgt"

rr = H.infer_compare_w_gt(
  gmodel,
  "$(gtdir)/Gauguin,_Paul_-__Three_Tahitian_Women_Against_a_Yellow_Background.jpg",
  "$(gtdir)/Gauguin,_Paul_-__Three_Tahitian_Women_Against_a_Yellow_Background-mask.jpg",
)



imshow(rr.overlaid)


rr = H.infer_compare_w_gt(
  gmodel,
  "$(gtdir)/Derain,_Andre_-_Port.jpg",
  "$(gtdir)/Derain,_Andre_-_Port-mask.jpg"
)

imshow(rr.overlaid)

# metrics = H.eval_on_image_dir(gmodel, "/home/pmilovanov/data/hair/private/realgt")

# @show pairs(metrics)
