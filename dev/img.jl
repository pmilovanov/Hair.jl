using Images, ImageView
using Hair




imneg(img) = 1 .- img


hairmos(img, comps, i, ncolumns) = imshow(Hair.compmosaic(
    img,
    comps[i:(i+ncolumns*ncolumns-1)],
    ncol = ncolumns,
    nrow = ncolumns,
    fillvalue = Gray{N0f8}(1.0)
))

img = load(ENV["HOME"] * "/data/hair/scratch/cropped.png")
#img= load(ENV["HOME"] * "/data/hair/scans/out0014.png")
#img = img0

mask95 = (img .< 0.9)
#mask95_o = diameter_opening(mask95, min_diameter=20)


comps = Hair.components(mask95, minarea=50)[2:end]


hairmos(img, comps, 3, 5)
