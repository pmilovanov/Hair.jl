using ImageMagick

#! format: off
"""
Read files from dir, output (filename, file_contents) tuples to channel.
Closes channel after reading.

Note: filename returned is a basename, not a full path.
"""
function read_from_dir(c::Channel, path::AbstractString)
  try for f=readdir(path, join=true); put!(c, (basename(f), read(f))); end
  catch e; @error e; rethrow(e)
  finally close(c); end
end

"""
Read images and corresponding masks as binary blobs from a dir to a channel.
Returns (filename, image, mask) tuple.
"""
function read_images_masks(out::Channel, path::AbstractString="", filenames::Union{Array{AbstractString,1},Nothing}=nothing; mask_suffix::AbstractString="-mask")
  try
    if filenames==nothing
      filenames = [x for x in readdir(path) if !contains(x,mask_suffix)]
    end
    for f in filenames
      noextname,ext = splitext(f)
      maskfname = joinpath(path, noextname*mask_suffix*ext)
      fname = joinpath(path, f)
      put!(out, (f, read(fname), read(maskfname)))
    end
  catch e; @error e;
  finally
    close(out)
  end
end

read_images_masks_from_dir(out::Channel, path::AbstractString, mask_suffix::AbstractString="-mask") = read_images_masks(out, path; mask_suffix=mask_suffix)

"""
Decode image, mask binary blobs.

- input: channel of (filename, image_blob, mask_blob) tuples
- output: channel of (filename, image, mask) tuples
"""
function load_images_masks(input::Channel, output::Channel)
  try
    for (fname, img, mask)=input
      put!(output, (fname, RGB.(readblob(img)), readblob(mask)))
    end
  catch e; @error e;
  finally; close(output); end
end


#! format: on
