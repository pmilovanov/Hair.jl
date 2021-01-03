# Functions to generate training data using stuff from img.jl

abstract type SamplingStrategy end

sample_image(img::Image{T}, s::S)::Array{Image{T},1} where {T, S<:SamplingStrategy} = error("Not implemented")

struct GridStrategy <: SamplingStrategy
    nsamples::Int
    sidelength::Int
    overlap::Int
end


function sample_image(img::Image, s::GridStrategy)
    if any(size(img) .< s.sidelength); return []; end
    
    times_fully_fits_into_length(l_given, l_into, overlap) = ((l_into - l_given)÷overlap + 1)
    nfits_x = times_fully_fits_into_length(size(img)[1], s.sidelength, s.overlap)
    nfits_y = times_fully_fits_into_length(size(img)[2], s.sidelength, s.overlap)

    candidates = [(1,1) .+ (s.overlap*(i-1), s.overlap*(j-1)) for i=1:nfits_x for j=1:nfits_y]
    samples = candidates
    
    if nsamples != length(candidates)
        # sample from candidates
    end

    [img[c[1]:c[1]+s.sidelength-1, c[2]:c[2]+s.sidelength-1] for c in samples]
end


struct RandomPositionStrategy <: SamplingStrategy
    nsamples::Int
    sidelength::Int
end



    
