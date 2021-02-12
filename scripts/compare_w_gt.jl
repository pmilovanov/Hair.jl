include("loadenv.jl")

@load "/home/pmilovanov/data/hair/models/fav/best.bson" model
gmodel = gpu(model)

gtdir = "/home/pmilovanov/data/hair/private/realgt"

# rr = H.infer_compare_w_gt(
#   gmodel,
#   "$(gtdir)/Gauguin,_Paul_-__Three_Tahitian_Women_Against_a_Yellow_Background.jpg",
#   "$(gtdir)/Gauguin,_Paul_-__Three_Tahitian_Women_Against_a_Yellow_Background-mask.jpg",
# )


metrics = H.eval_on_image_dir(gmodel, "/home/pmilovanov/data/hair/private/realgt")

@show pairs(metrics)
