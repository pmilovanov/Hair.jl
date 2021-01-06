using Images, ImageView
using Hair
H = Hair



imneg(img) = 1 .- img


hairmos(img, comps, i, ncolumns) =
  imshow(Hair.compmosaic(img, comps[i:(i+ncolumns*ncolumns-1)], ncol = ncolumns, nrow = ncolumns, fillvalue = Gray{N0f8}(1.0)))

hair_img = load(ENV["HOME"] * "/data/hair/scratch/cropped.png")
#img= load(ENV["HOME"] * "/data/hair/scans/out0014.png")
#img = img0


gopher = load("/home/pmilovanov/tmp/gopher.jpeg")

#hairmos(img, comps, 3, 5)
hairs = H.gen_single_hairs(hair_img)
hair = hairs[12]

#imshow(H.place(hair, gopher, (500, 500), H.multiply_luminance))
imshow(hair)

hair_wa = H.matte_with_color(hair)
go1 = copy(gopher)

H.place!(hair_wa, go1, (100,100), H.ontop)

H.place!(hair_wa, go1, (250,100), H.multiply_luminance)

H.place!(imresize(hair_wa, ratio=0.5), go1, (200,200), H.ontop)


imshow(go1)
