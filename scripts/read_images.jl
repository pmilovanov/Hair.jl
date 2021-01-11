using Images, ImageMagick
using Base.Threads: @spawn, threadid, @threads, nthreads
using ProgressMeter

@info "Nthreads: $(nthreads())"

function par_load_img()
  c_bindata = Channel(1000)
  c_images = Channel(1000)

  path = "/home/pmilovanov/data/hair/hairy/exp/0107-01"
  # path = "/home/pmilovanov/data/the_met/temp/oil_sample_100"
  fnames = readdir(path, join = true)[1:1000]
  @async begin
    for f in fnames
      try
        put!(c_bindata, read(f))
      catch e
        @error e
      end
    end
    close(c_bindata)
  end

  @spawn begin
    @sync for blob in c_bindata
      @spawn try
        put!(c_images, readblob(blob))
      catch e
        @error e
      end
    end
    close(c_images)
  end

  v = Vector{Any}()
  p = Progress(length(fnames), desc = "Loading images")
  for img in c_images
    push!(v, img)
    next!(p)
    flush(stderr)
  end

  println(Base.summarysize(v) / (2^20))
end


@time par_load_img()
