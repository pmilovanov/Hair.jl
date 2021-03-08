using Mocking


const GCS_PREFIX = "gs://"

isgcs(path::String) = startswith(path, GCS_PREFIX)

struct GCSPath
  path::String
  function GCSPath(path::String)
    if isgcs(path)
      path = path[length(GCS_PREFIX)+1:end]
    end
    new(path)
  end
end

macro gs_str(x)
  :(GCSPath($x))
end

gcscopy(src::String, dest::String) = @mock run(`gsutil -m cp "$(src)" "$(dest)"`)

cp(src::GCSPath, dest::String) = gcscopy(uri(src), dest)
cp(src::String, dest::GCSPath) = gcscopy(src, uri(dest))

uri(p::GCSPath) = "$(GCS_PREFIX)$(p.path)"

function untar(src::GCSPath, destdir::String)
  !isdir(destdir) && raise(ArgumentError("Directory $destdir doesn't exist or isn't a directory"))
  !endswith(src.path, ".tar") && raise(ArgumentError("Source must be a tar file"))
  curdir = pwd()
  cd(destdir)
  gcscopy(uri(src), joinpath(destdir, ""))
  bname = basename(src.path)
  run(`tar xf $(bname)`)
  run(`rm $(bname)`)
  cd(curdir)
end

function downloadmemaybe(filename::String)
  if !isgcs(filename); return filename; end
  localfname = joinpath(mktempdir(), "model.bson")
  gcscopy(filename, localfname)
  localfname
end

function gsls(path::String)
  if !isgcs(pattern); return readdir(path, join=true); end
  return open(`gsutil ls "$path"`, stdout; read=true) do io
    lines = Vector{String}
    while !eof(io)
      push!(lines, readline(io))
    end
    lines
  end
end

