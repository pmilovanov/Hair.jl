using ArgParse, StatsBase, ProgressMeter



#! format=off
function parse_flags()
  s = ArgParseSettings()
  @add_arg_table! s begin
    "--inputdir"
      required = true
    "--outputdir"
      required = true
    "--nimages", "-n"
      arg_type = Int
      default = 10
    "--prefix"
      default = nothing
  end
  
  args = parse_args(s)
  for k in ["inputdir", "outputdir"]
    args[k] = expanduser(args[k])
  end
  args
end
#! format=off

function main(;inputdir::String, outputdir::String, nimages::Int, prefix=nothing)

  outputdir = expanduser(outputdir)
  if !isdir(outputdir); mkpath(outputdir); end

  p = Progress(nimages)
  fnames = [fname for fname in readdir(inputdir, join=true) if (contains(fname, ".jpg") || contains(fname, ".jpeg"))]
  for src in sample(fnames, nimages, replace=false)
    dest = basename(src)
    if prefix != nothing; dest = "$(prefix)_$(dest)"; end
    cp(src, joinpath(outputdir, dest))
    next!(p)
  end

end

tosymboldict(d) = Dict([Symbol(k)=>v for (k,v) in d])

@time main(;tosymboldict(parse_flags())...)
