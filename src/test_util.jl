module TestUtil

using Printf, Images

function write_dummy_images_masks(N::Int, imgsize::Int)
  tdir = mktempdir()
  A = rand(N, 4, imgsize, imgsize)
  imgs = [RGB{N0f8}.(A[i, 1, :, :], A[i, 2, :, :], A[i, 3, :, :]) for i = 1:N]
  masks = [(A[i, 4, :, :] .> 0.9) for i = 1:N]
  for i = 1:N
    save(joinpath(tdir, @sprintf("%d.png", i)), imgs[i])
    save(joinpath(tdir, @sprintf("%d-mask.png", i)), masks[i])
  end
  return tdir
end

end
