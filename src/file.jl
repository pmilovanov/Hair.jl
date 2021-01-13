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

function read_images_masks_from_dir(out::Channel, path::AbstractString, mask_suffix::AbstractString="-mask")
  try
    fnames = [x for x in readdir(path) if !contains(x,mask_suffix)]
    for f in fnames
      noextname,ext = splitext(f)
      maskfname = joinpath(path, noextname*mask_suffix*ext)
      fname = joinpath(path, f)
      put!(out, (read(fname), read(maskfname)))
    end
  catch e
    @error e; rethrow(e)
  finally
    close(out)
  end
end

function load_images_masks(input::Channel, output::Channel)
  try
    for (img, mask)=input
      put!(output, (readblob(img), readblob(mask)))
    end
  catch e; @error e; rethrow(e)
  finally; close(output); end
end


#! format: on
