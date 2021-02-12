using Base.Iterators: partition
using Flux.Data: DataLoader
using Parameters: @with_kw
using Flux, CUDA
using Flux: throttle, logitbinarycrossentropy
using Statistics
using Printf
using MLDataPattern: splitobs, shuffleobs
using ProgressMeter
using BSON: @load, @save
import Dates
using NNlib

function infer_old(model, image::Image{RGB{Float32}}; inference_side::Int = 1024, device = gpu)
  model = device(model)
  newsize = convert.(Int, ceil.(size(image) ./ inference_side)) .* inference_side
  needresize = (size(image) != newsize)
  newimg = needresize ? imresize(image, newsize) : image

  step::Int = inference_side ÷ 2
  h = size(image)[1] / step - 1
  w = size(image)[2] / step - 1

  Ys = zeros(Float32, newsize...)

  for i = 1:h
    for j = 1:w
      (x0, x1) = convert.(Int, (i - 1, i + 1) .* step .+ (1, 0))
      (y0, y1) = convert.(Int, (j - 1, j + 1) .* step .+ (1, 0))

      X = device(imgtoarray(newimg[x0:x1, y0:y1]))
      Y = cpu(model(X))
      Ys[x0:x1, y0:y1] .= max.(Ys[x0:x1, y0:y1], Y[:, :, 1, 1])

      #Ys[x0:x1, y0:y1] .= Y[inference_side, inference_side, 1, 1]
    end
  end

  outimg = Gray{Float32}.(Ys)
  return needresize ? imresize(outimg, size(image)) : outimg
end

function infer(
  model,
  image::Image{RGB{Float32}};
  inference_side::Int = 1024,
  overlap = 512,
  batchsize = 4,
  device = gpu,
)
  model = device(model)

  step::Int = inference_side ÷ 2
  h = size(image)[1] / step - 1
  w = size(image)[2] / step - 1

  Ys = zeros(Float32, size(image)...)

  subimages = sample_image_w_coords(
    image,
    GridStrategy(side = inference_side, overlap = overlap, cover_edges = true),
  )

  while true
    done = false
    coords = zeros(Int, batchsize, 2)
    X = zeros(Float32, inference_side, inference_side, 3, batchsize)
    try
      for i = 1:batchsize
        subimage = pop!(subimages)
        coords[i, :] .= first(subimage)
        X[:, :, :, i] .= imgtoarray(last(subimage))[:, :, :, 1]
      end
    catch e
      @assert isa(e, ArgumentError)
      done = true
    end

    X = X |> device
    Y = model(X) |> cpu

    for i = 1:batchsize
      x0, y0 = coords[i, :]
      if (x0, y0) == (0, 0)
        break
      end
      x1, y1 = (x0, y0) .+ inference_side .- 1
      Ys[x0:x1, y0:y1] .= max.(Ys[x0:x1, y0:y1], Y[:, :, 1, i])
    end

    if done
      break
    end
  end

  return Gray{Float32}.(Ys)
end



function infer(model, path::String; kwargs...)
  img = RGB{Float32}.(load(path))
  (img, infer(model, img; kwargs...))
end


function eval_on_image(model, input_path::String, gt_path::String, resize_ratio::Real = 1.0)
  ŷ = infer(model, input_path) |> last
  y = convert.(Gray{Float32}, load(gt_path))

  if resize_ratio != 1.0
    ŷ = imresize(ŷ, ratio = resize_ratio)
    y = imresize(y, ratio = resize_ratio)
  end  

  prf1(channelview(ŷ), channelview(y))
end


GrayImage = Matrix{Gray{Float32}}
function eval_on_images(images::Vector{NTuple{2, GrayImage}})

  accum = values(BinarySegmentationMetrics(zeros(Int, 10)))

  for (ŷ, y) in images
    accum = accum .+ values(binsegmetrics(channelview(ŷ), channelview(y)))
  end

  a = BinarySegmentationMetrics(accum)

  p = a.tp / (a.tp + a.fp)
  r = a.tp / (a.tp + a.fn)
  f1 = 2*p*r / (p+r)

  BinarySegmentationMetrics( (p, r, f1, a.ap_ŷ, a.ap_y, a.tp, a.tn, a.fp, a.fn, a.npixels) )
end

function eval_on_image_dir(model, image_dir::String)
  files = [x for x in readdir(expanduser(image_dir), join=true) if endswith(x, "-mask.jpg")]

  images = Vector{NTuple{2, GrayImage}}()
  for mask_fname in files
    @info mask_fname
    img_fname = replace(mask_fname, "-mask.jpg"=>".jpg")

    img, ŷ = infer(model, img_fname)
    y = convert.(Gray{Float32}, load(mask_fname))
    push!(images, (ŷ, y))
  end
  eval_on_images(images)
end

struct InferenceVisualComparisonResult
  img::Image{RGB{Float32}}
  ŷ::Image{Gray{Float32}}
  y::Image{Gray{Float32}}
  overlaid::Image{RGB{Float32}}
  metrics::Dict{Symbol, Float32}
  eval_resize_ratio::Float64
end

function infer_compare_w_gt(model, input_path::String, gt_path::String; resize_ratio::Float64 = 1.0)
  img, ŷ = infer(model, input_path)
  y = convert.(Gray{Float32}, load(gt_path))

  ŷr, yr = ŷ, y
  if resize_ratio != 1.0
    ŷr = imresize(ŷr, ratio=resize_ratio)
    yr = imresize(yr, ratio=resize_ratio)
  end
  precision, recall, f1 = prf1(channelview(ŷr), channelview(yr))

  oimg = deepcopy(img)
  place_overlay!(oimg, ŷ, channel = 1)
  place_overlay!(oimg, y, channel = 2)

  InferenceVisualComparisonResult(img, ŷ, y, oimg, Dict(:precision=>precision, :recall=>recall, :f1=>f1), resize_ratio)
end



function place_overlay!(
  img::Image{T} where {T},
  overlay::Image{Gray{G}} where {G},
  mask_threshold = 0.1,
  outline = true;
  channel = 1,
)
  rimg = rawview(channelview(img))

  if outline == true
    binmask = overlay .> mask_threshold
    for i = 1:7
      dilate!(binmask)
    end

    outline = copy(binmask)
    for i = 1:3
      dilate!(outline)
    end

    outline = outline .* (.!binmask)
    overlay = convert.(eltype(overlay), outline)
  end

  roverlay = rawview(channelview(overlay))
  rimg[channel, :, :] = max.(rimg[channel, :, :], roverlay)

  img
end



function infer_overlay(model, path::String, side::Int = 1024, mask_threshold = 0.1; kwargs...)
  img, mask = infer(model, path; kwargs...)
  @show size(mask)
  @show size(img)
  mask = Gray.(mask .> mask_threshold)

  overlay = place_overlay!(copy(img), mask)
  return (img, overlay)
end



function latestmodel(; dir = expanduser("~/data/hair/models"), modelid = nothing, epoch = 0)
  modeldir =
    (modelid == nothing) ? sort(readdir(expanduser(dir), join = true), rev = true)[1] :
    joinpath(dir, modelid)
  return (epoch == 0) ? sort(readdir(modeldir, join = true), rev = true)[1] :
         joinpath(modeldir, @sprintf("epoch_%03d.bson", epoch))
end
