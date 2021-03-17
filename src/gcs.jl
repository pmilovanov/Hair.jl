using Mocking


const GCS_PREFIX = "gs://"

struct GCSError <: Exception
  message::String
end

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
  if !isgcs(filename)
    return filename
  end
  localfname = joinpath(mktempdir(), "model.bson")
  gcscopy(filename, localfname)
  localfname
end

function readlines(cmd::Cmd, default = Vector{String}())
  try
    return @mock open(cmd, stdout; read = true) do io
      lines = Vector{String}()
      while !eof(io)
        push!(lines, readline(io))
      end
      lines
    end
  catch
    return default
  end
end

gsls(path::String) = isgcs(path) ? readlines(`gsutil ls $path`) : readdir(path, join = true)

function gsisdir(path::String)
  isgcs(path) || return isdir(path)
  lines = readlines(`gsutil ls -d $path`)
  return length(lines) == 1 && lines[1][end] == '/'
end

function gsisdirwfiles(path::String)
  if isgcs(path)
    lines = readlines(`gsutil ls $path`)
    length(lines) == 0 && return false
    length(lines) > 1 && return true
    return lines[1] != path
  end

  return ispath(path) && isdir(path) && length(readdir(path)) > 0
end


@enum PathType begin
  PATH_NONEXISTENT
  PATH_FILE
  PATH_DIR_EMPTY
  PATH_DIR_NONEMPTY
  PATH_OTHER
end

function localpathtype(path::String)::PathType
  !ispath(path) && return PATH_NONEXISTENT
  isfile(path) && return PATH_FILE
  isdir(path) && return length(readdir(path)) > 0 ? PATH_DIR_NONEMPTY : PATH_DIR_EMPTY
  PATH_OTHER
end

function gsonlypathtype(path::String)::PathType
  lines = readlines(`gsutil ls $path`)
  length(lines) == 0 && return PATH_NONEXISTENT
  length(lines) > 1 && return PATH_DIR_NONEMPTY
  lines[1] == path && return PATH_FILE
  throw(GCSError("Can't have empty dirs in gcs but found '$(lines[1])' for query '$(path)'"))
end

function gspathtype(path::String)::PathType
  return isgcs(path) ? gsonlypathtype(path) : localpathtype(path)
end

function gsmkpath(path::String)
  if !isgcs(path)
    mkpath(path)
  end
end
