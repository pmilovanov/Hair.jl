# Functions to generate training data using stuff from img.jl

using StatsBase
using Base: @kwdef

abstract type SamplingStrategy end

sample_image(img::Image{T}, s::S) where {T,S<:SamplingStrategy} = error("Not implemented")

@kwdef struct GridStrategy <: SamplingStrategy
  n::Int = 0
  side::Int = 128
  overlap::Int = 10
  cover_edges::Bool = true
end


function sample_image(img::Image, s::GridStrategy)
  if any(size(img) .< s.side)
    return []
  end

  times_fully_fits_into_length(l_into, l_given, overlap) = ((l_into - l_given) ÷ (l_given - overlap) + 1)
  nfits_x = times_fully_fits_into_length(size(img)[1], s.side, s.overlap)
  nfits_y = times_fully_fits_into_length(size(img)[2], s.side, s.overlap)

  step = s.side - s.overlap
  candidates = [(1, 1) .+ (step * (i - 1), step * (j - 1)) for j = 1:nfits_y for i = 1:nfits_x]

  if s.cover_edges
    cover_x = ((step * (nfits_x - 1) + s.side) != size(img)[1])
    cover_y = ((step * (nfits_y - 1) + s.side) != size(img)[2])
    if cover_x
      push!(candidates, [(size(img)[1] - s.side + 1, 1 + step * (i - 1)) for i = 1:nfits_y]...)
    end
    if cover_y
      push!(candidates, [(1 + step * (i - 1), size(img)[2] - s.side + 1) for i = 1:nfits_x]...)
    end
    if cover_x && cover_y
      push!(candidates, (size(img)[1] - s.side + 1, size(img)[2] - s.side + 1))
    end
  end

  samples = (s.n == 0 || s.n == length(candidates)) ? candidates : [sample(candidates) for i = 1:s.n]

  [img[c[1]:c[1]+s.side-1, c[2]:c[2]+s.side-1] for c in samples]
end

function gen_single_hairs(img::Image; threshold::Float64 = 0.9, minhairarea::Int = 50, n::Int = 1000)
  hsl = HSL.(color.(img))
  lum = comp3.(hsl)
  mask = (lum .< threshold)
  comps = components(mask, minarea = minhairarea)[2:end]
  matte_color = mode(img)
  [matte_with_color(image(img, comp), matte_color) for comp in comps[1:min(n, length(comps))]]
end



function t_random(;scale=(0.25,1), θ=(0,2π), opacity=(0.3, 1))
  randrange(rmin, rmax) = rand()*(rmax-rmin) + rmin
  function transform(img::Image)
    img = imrotate(img, randrange(θ...))
    img = imresize(img, ratio=randrange(scale...))
  end
end

                                              


function put_hairs(dest, n::Int, allhairs::Array{Image{T},1} where {T}, transform_fn)
  dest = copy(dest)
  for i = 1:n
    img = transform_fn(sample(allhairs))
    x0, y0, x1, y1 = (1, 1, size(dest)...) .- (size(img)..., size(img)...)
    pos = sample(x0:x1), sample(y0:y1)
    place!(img, dest, pos)
  end
  dest
end
