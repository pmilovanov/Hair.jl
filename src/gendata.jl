# Functions to generate training data using stuff from img.jl

using Base.Threads: @spawn
using Parameters: @with_kw
using Images
using Printf
using ProgressMeter
using Random: shuffle!
using StatsBase

abstract type SamplingStrategy end

sample_image(img::Image{T}, s::S) where {T,S<:SamplingStrategy} = error("Not implemented")

@with_kw struct GridStrategy <: SamplingStrategy
  n::Int = 0
  side::Int = 128
  overlap::Int = 10
  cover_edges::Bool = true
end


function sample_image_w_coords(img::Image, s::GridStrategy)
  any(size(img) .< s.side) && return []

  times_fully_fits_into_length(l_into, l_given, overlap) =
    ((l_into - l_given) ÷ (l_given - overlap) + 1)
  nfits_x = times_fully_fits_into_length(size(img)[1], s.side, s.overlap)
  nfits_y = times_fully_fits_into_length(size(img)[2], s.side, s.overlap)

  step = s.side - s.overlap
  candidates = [(1, 1) .+ (step * (i - 1), step * (j - 1)) for j = 1:nfits_y for i = 1:nfits_x]

  if s.cover_edges
    cover_x = ((step * (nfits_x - 1) + s.side) != size(img)[1])
    cover_y = ((step * (nfits_y - 1) + s.side) != size(img)[2])
    cover_x &&
      push!(candidates, [(size(img)[1] - s.side + 1, 1 + step * (i - 1)) for i = 1:nfits_y]...)
    cover_y &&
      push!(candidates, [(1 + step * (i - 1), size(img)[2] - s.side + 1) for i = 1:nfits_x]...)
    cover_x && cover_y && push!(candidates, (size(img)[1] - s.side + 1, size(img)[2] - s.side + 1))
  end

  samples =
    (s.n == 0 || s.n == length(candidates)) ? candidates : [sample(candidates) for i = 1:s.n]

  [(c, @view(img[c[1]:c[1]+s.side-1, c[2]:c[2]+s.side-1])) for c in samples]
end

sample_image(img::Image, s::GridStrategy) = [last(c) for c in sample_image_w_coords(img, s)]

function gen_single_hairs(
  img::Image;
  threshold::Float64 = 0.9,
  minhairarea::Int = 50,
  n::Int = 1000,
)
  hsl = HSL.(color.(img))
  lum = comp3.(hsl)
  mask = (lum .< threshold)
  comps = components(mask, minarea = minhairarea)[2:end]
  matte_color = mode(img)
  [matte_with_color(image(img, comp), matte_color) for comp in comps[1:min(n, length(comps))]]
end


iminvert(img::Image{C}) where {C<:Color} = convert(eltype(img), 1.0) .- img
iminvert(img::Image{TC}) where {TC<:TransparentColor} =
  coloralpha.(iminvert(color.(img)), alpha.(img))


function imblur(
  img::Image{TC},
  σ::AbstractFloat,
  kernel_size::Int = 9;
  pad::Bool = true,
) where {TC<:Colorant}
  img = padarray(img, Fill(zero(TC), (kernel_size, kernel_size), (kernel_size, kernel_size)))
  k = centered(Kernel.gaussian([σ, σ], [kernel_size, kernel_size]))
  imfilter(img, k)
end

@with_kw struct RandomizedTransformParams{T<:AbstractFloat}
  scale::Tuple{T,T} = (0.25, 1)
  θ::Tuple{T,T} = (0, 2π)
  opacity::Tuple{T,T} = (0.3, 1)
  invert_prob::T = 0.2

  blur_prob::T = 0.4
  blur_max_σ::T = 20.0
  blur_kernel_size::Int = 9

  gradient_prob::T = 0.1
  gradient_θ::Tuple{T,T} = (0, 2π)
end

function t_random(p::RandomizedTransformParams = RandomizedTransformParams{Float64}())
  randrange(rmin, rmax) = rand() * (rmax - rmin) + rmin

  function transform(img::Image{TC}) where {TC<:TransparentColor}
    imgcolor, imgα = color.(img), alpha.(img)
    imgα = convert.(eltype(eltype(img)), imgα .* randrange(p.opacity...))

    imgcolor = rand() < p.invert_prob ? iminvert(imgcolor) : imgcolor
    img = coloralpha.(imgcolor, imgα)

    #img = rand() < p.blur_prob ? imblur(img, rand()*p.blur_max_σ, p.blur_kernel_size) : img
    img = imrotate(img, randrange(p.θ...))
    img = imresize(img, ratio = randrange(p.scale...))
  end
end

function put_hairs(dest::Image, n::Int, allhairs::Array{<:Image{T},1} where {T}, transform_fn)
  dest = copy(dest)
  mask = zeros(Gray{eltype(eltype(dest))}, size(dest))

  for i = 1:n
    img = transform_fn(sample(allhairs))
    x0, y0, x1, y1 = (1, 1, size(dest)...) .- (size(img)..., size(img)...)
    pos = sample(x0:x1), sample(y0:y1)
    place!(img, dest, pos)
    place!(convert.(Gray, alpha.(img)), mask, pos, (a, b) -> max(a, b))
  end
  mask = mask .> 0.05
  dest, mask
end

@with_kw struct MakeHairySquaresOptions
  samples_per_pic::Int = 20
  square_size::Int = 512
  prob_any_hairs::Float64 = 0.9
  max_hairs_per_output = 5
end

function sample_image_and_add_hairs(
  img::Image,
  hairs::Vector{<:Image{T}} where {T},
  o::MakeHairySquaresOptions = MakeHairySquaresOptions();
  img_id::Int = 0,
)
  out = Vector{Any}()
  samples = sample_image(
    img,
    GridStrategy(n = o.samples_per_pic, side = o.square_size, overlap = o.square_size ÷ 3),
  )
  for (j, sample) in enumerate(samples)
    numhairs = (rand() < o.prob_any_hairs) ? rand(1:o.max_hairs_per_output) : 0
    sample, mask = put_hairs(sample, numhairs, hairs, t_random())
    push!(out, (img_id, j, sample, mask))
  end
  out
end

function make_hairy_squares(
  hairs::Vector{<:Image{T}} where {T},
  pics_dir::AbstractString,
  output_dir::AbstractString,
  o::MakeHairySquaresOptions = MakeHairySquaresOptions(),
)
  N_CHANNEL_PICS = 128
  c_pics = Channel(N_CHANNEL_PICS)
  c_outputs = Channel(N_CHANNEL_PICS * o.samples_per_pic)
  c_progress = Channel(N_CHANNEL_PICS * o.samples_per_pic * 2)

  pics_fnames = shuffle(readdir(pics_dir))
  @info "Found $(length(pics_fnames)) images"

  @spawnlog c_pics for (i, fname) in enumerate(pics_fnames)
    put!(c_pics, (i, load(joinpath(pics_dir, fname))))
  end

  @spawnlog c_outputs begin
    @sync for (i, img) in c_pics
      @spawnlog for s in sample_image_and_add_hairs(img, hairs, o, img_id = i)
        put!(c_outputs, s)
      end
    end
  end

  @spawnlog c_progress for (i, j, sample, mask) in c_outputs
    @spawnlog begin
      basename = @sprintf("%06d-%06d", i, j)
      fname, mask_fname = basename * ".jpg", basename * "-mask.jpg"
      save(joinpath(output_dir, fname), sample)
      save(joinpath(output_dir, mask_fname), mask)
      put!(c_progress, 1)
    end
  end

  noutputs = length(pics_fnames) * o.samples_per_pic

  p = Progress(noutputs)
  for _ in c_progress
    next!(p)
  end
end
