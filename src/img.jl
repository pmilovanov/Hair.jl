using Images, ImageDraw
import Base.isless
using MosaicViews

# Bounding box
const BBox = Array{Tuple{Int64,Int64},1}

struct Component
    index::Int
    length::Int
    bbox::BBox
    mask::BitArray{2}
end

function boxpoly(c::Component)
    (x0, y0), (x1, y1) = c.bbox
    Polygon([Point(x0, y0), Point(x0, y1), Point(x1, y1), Point(x1, y0)])
end

function area(box::BBox)
    (x0, y0), (x1, y1) = box
    abs((x1 - x0) * (y1 - y0))
end

Base.isless(x::Component, y::Component) = area(x.bbox) < area(y.bbox)


function imgslice(img, bbox::BBox)
    (x0, y0), (x1, y1) = bbox
    img[x0:x1, y0:y1]
end

function components(img::BitArray{2}; minarea::Int=0)
    cc = label_components(img)
    lengths = component_lengths(cc)
    boxes = component_boxes(cc)
    results = sort(
        [
            Component(i, len_i, box_i, imgslice(cc, box_i) .== i-1)
            for (i, (len_i, box_i)) in enumerate(zip(lengths, boxes))
            if len_i > minarea
        ],
        rev = true,
    )[2:end]
end

imneg(img) = 1 .- img

#image(img::Array{T,2}, component::Component) where {T} =
#    max.(imneg(component.mask), imgslice(img, component.bbox))

image(img::Array{T,2}, c::Component) where {T} =
    coloralpha.(imgslice(img, c.bbox), c.mask)


compmosaic(img, components::Array{Component,1}; kwargs...) =
    mosaicview([image(img, c) for c in components]; kwargs...)

function opening_n!(img, n)
    for i in 1:n; dilate!(img); end
    for i in 1:n; erode!(img); end
end

const K = Kernel
function anglegrad(img::Array{T, 2}, angle::Float64) where {T}
    gx,gy = imgradients(img, K.ando5)
    cos(angle).*gx + sin(angle).*gy
end
    
