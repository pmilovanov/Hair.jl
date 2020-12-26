using Images, ImageDraw, MosaicViews
import Base.isless

"Bounding box"
const BBox = Array{Tuple{Int64,Int64},1}
bbox(x1, y1, x2, y2) = [(x1, y1), (x2, y2)]
x1(b::BBox) = b[1][1]
y1(b::BBox) = b[1][2]
x2(b::BBox) = b[2][1]
y2(b::BBox) = b[2][2]

const Image{T} = Array{T,2} where {T}
const ImageMask = BitArray{2}

struct Component
    index::Int
    length::Int
    bbox::BBox
    mask::BitArray{2}
end

"Make a polygon out of a bounding box produced by functions like `label_components(...)`"
boxpoly(b::BBox) = Polygon([
    Point(x1(b), y1(b)),
    Point(x1(b), y2(b)),
    Point(x2(b), y2(b)),
    Point(x2(b), y1(b)),
])
boxpoly(c::Component) = boxpoly.bbox

area(b::BBox) = abs((x2(b) - x1(b)) * (y2(b) - y1(b)))

Base.isless(x::Component, y::Component) = area(x.bbox) < area(y.bbox)


imgslice(img::Image{T}, b::BBox) where {T} = img[x1(b):x2(b), y1(b):y2(b)]

function components(img::ImageMask; minarea::Int = 0)
    cc = label_components(img)
    lengths = component_lengths(cc)
    boxes = component_boxes(cc)
    results = sort(
        [
            Component(i, len_i, box_i, imgslice(cc, box_i) .== i - 1)
            for
            (i, (len_i, box_i)) in enumerate(zip(lengths, boxes)) if len_i > minarea
        ],
        rev = true,
    )[2:end]
end

imneg(img::Image{T}) where {T} = 1 .- img

#image(img::Array{T,2}, component::Component) where {T} =
#    max.(imneg(component.mask), imgslice(img, component.bbox))

image(img::Image{T}, c::Component) where {T} = coloralpha.(imgslice(img, c.bbox), c.mask)

compmosaic(img::Image{T}, components::Array{Component,1}; kwargs...) where {T} =
    mosaicview([image(img, c) for c in components]; kwargs...)

function opening_n!(img::Image{T}, n::Int) where {T}
    for i = 1:n
        dilate!(img)
    end
    for i = 1:n
        erode!(img)
    end
end

"Gradient of an image in the direction specified by the angle θ"
function anglegrad(img::Array{T,2}, θ::Float64) where {T}
    gx, gy = imgradients(img, Kernel.ando5)
    cos(θ) .* gx + sin(θ) .* gy
end


function srcdestboxes(
    srcsize::Tuple{Int,Int},
    destsize::Tuple{Int,Int},
    topleft::Tuple{Int,Int},
)
    @assert all(srcsize .> 0)
    @assert all(destsize .> 0)
    
    btmright = topleft .+ srcsize
    if any(btmright .< 1) || any(btmright .> destsize)
        return [(0, 0), (0, 0)], [(0, 0), (0, 0)]
    end

    real_btmright = min.(destsize, btmright)
    real_topleft = max.((1, 1), topleft)

    destbox = [real_topleft, real_btmright]
    srcbox = [real_topleft .- topleft .+ (1, 1), real_btmright .- topleft .+ (1, 1)]

    srcbox, destbox
end

"""
Place image `img` on top of image `dest` with `img`'s top left corner at location `(x,y)`
relative to the destination image.

- Returned image's size is the same as the destination image and has the same color type.
- `img` has alpha channel (transparency), `dest` does not.
- `img`'s non-transparent color type must be convertible to `dest`'s color type.
"""
function place(
    img::Image{Transparent3{T}},
    dest::Image{Color3{T}},
    x::Int,
    y::Int,
) where {T}

end
