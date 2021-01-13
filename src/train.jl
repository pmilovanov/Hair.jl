

function load_data(datadir::String; bufsize::Int=256)
  fnames = readdir(datadir, join=true)
  nimgs = length(fnames)/2; @assert !contains(fnames[1], "-mask")
  w = size(load(fnames[1]))[1]

  imgs = Array{Float32, 4}(undef, w, w, 3, nimgs)
  masks = Array{Float32, 4}(undef, w, w, 1, nimgs)

  c_blobs = Channel(bufsize)
  c_imgs = Channel(bufsize)
  @async H.read_images_masks_from_dir(c_blobs, tdir)
  @async H.load_images_masks(c_blobs, c_imgs)
  
  for (i, (img, mask)) = enumerate(c_imgs)
    imgs[:,:,:,i] = Float32.(permutedims(channelview(img), [2,3,1]))
    mask[:,:,1,i] = Float32.(mask .> 0.9)
  end

  imgs, masks
end
