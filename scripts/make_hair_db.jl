using ArgParse, Hair, ProgressMeter, JLD2, Images, Distributed
import Base.Threads.@spawn

H = Hair

function parse_flags()
  s = ArgParseSettings()
  @add_arg_table! s begin
    "--input_dir"
    help = "Dir with pngs of images of hair on white paper"
    "--output"
    help = "Output to this JLD2 file"
  end
  args = parse_args(s)
  for k in keys(args)
    args[k] = expanduser(args[k])
  end
  args
end

function main(input_dir, output)
  hairsets = []
  fnames = [fn for fn in readdir(input_dir, join = true) if endswith(fn, ".png")]

  p = Progress(length(fnames), 1)
  c = Channel(32)

  compute_hairs(ch::Channel, fname) = put!(c, H.gen_single_hairs(load(fname)))

  n = length(fnames)
  for fname in fnames
    @spawn compute_hairs(c, fname)
  end
  while n > 0
    push!(hairsets, take!(c))
    next!(p)
    n = n - 1
  end

  println("Writing file")
  hairs = cat(hairsets..., dims = 1)
  @save output hairs
end

args = parse_flags()

@time main(args["input_dir"], args["output"])
