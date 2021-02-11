using ImageMagick



"""
Read images and corresponding masks as binary blobs from a dir to a channel.
Outputs (filename, image, mask) tuple to the `out` channel.

Function DOES NOT CLOSE the `out` channel when done!
"""
function read_images_masks(
  out::AbstractChannel,
  path::AbstractString = "";
  filenames::Union{AbstractArray{S,1},Nothing} where {S<:AbstractString} = nothing,
  mask_suffix::AbstractString = "-mask",
)
  if filenames == nothing
    filenames = [x for x in readdir(path) if !contains(x, mask_suffix)]
  end
  for f in filenames
    noextname, ext = splitext(f)
    maskfname = joinpath(path, noextname * mask_suffix * ext)
    fname = joinpath(path, f)
    put!(out, (f, read(fname), read(maskfname)))
  end
end

read_images_masks_from_dir(
  out::AbstractChannel,
  path::AbstractString,
  mask_suffix::AbstractString = "-mask",
) = read_images_masks(out, path; mask_suffix = mask_suffix)

"""
Decode image, mask binary blobs to Matrix{Color} images.

Schedules actual decoding on separate threads but waits synchronously until all decoding is done.

- input: channel of (filename, image_blob, mask_blob) tuples
- output: channel of (filename, image, mask) tuples. Function DOES NOT CLOSE the output channel when done!
"""
function load_images_masks(input::AbstractChannel, output::AbstractChannel)
  @sync for (fname, img, mask) in input
    @spawn put!(output, (fname, RGB{N0f8}.(readblob(img)), Gray{N0f8}.(readblob(mask))))
  end
end
