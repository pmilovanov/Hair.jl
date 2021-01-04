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

    times_fully_fits_into_length(l_into, l_given, overlap) =
        ((l_into - l_given) ÷ (l_given - overlap)+1)
    nfits_x = times_fully_fits_into_length(size(img)[1], s.side, s.overlap)
    nfits_y = times_fully_fits_into_length(size(img)[2], s.side, s.overlap)

    step = s.side - s.overlap
    candidates =
        [(1, 1) .+ (step * (i - 1), step * (j - 1)) for j = 1:nfits_y for i = 1:nfits_x]

    if s.cover_edges        
        cover_x = ((step * (nfits_x - 1) + s.side) != size(img)[1])
        cover_y = ((step * (nfits_y - 1) + s.side) != size(img)[2])
        if cover_x
            push!(
                candidates,
                [(size(img)[1] - s.side + 1, 1 + step * (i - 1)) for i = 1:nfits_y]...,
            )
        end
        if cover_y
            push!(
                candidates,
                [(1 + step * (i - 1), size(img)[2] - s.side + 1) for i = 1:nfits_x]...,
            )
        end
        if cover_x && cover_y
            push!(candidates, (size(img)[1] - s.side + 1, size(img)[2] - s.side + 1))
        end
    end

    samples =
        (s.n == 0 || s.n == length(candidates)) ? candidates :
        [sample(candidates) for i = 1:s.n]


    [img[c[1]:c[1]+s.side-1, c[2]:c[2]+s.side-1] for c in samples]
end


struct RandomPositionStrategy <: SamplingStrategy
    n::Int
    side::Int
end
