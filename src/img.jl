using Images, MosaicViews
import Base.isless
import ImageDraw

const IDr = ImageDraw




const Point2 = Tuple{Int, Int}

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
boxpoly(b::BBox) = IDr.Polygon([
    IDr.Point(x1(b), y1(b)),
    IDr.Point(x1(b), y2(b)),
    IDr.Point(x2(b), y2(b)),
    IDr.Point(x2(b), y1(b)),
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

"""
For a bounding box and a point A, find the point closest to A inside the box with 4-way connectivity.

If A is inside the box, A itself will be returned. Otherwise a point on the boundary of the box will be returned.
"""
bbox_minmax_point(box::BBox, point::Tuple{Int, Int})::Tuple{Int, Int} =
    min.(max.(point, first(box)), last(box)) 

function interval_overlap(src::Tuple{Int, Int}, dest::Tuple{Int, Int})::Tuple{Point2, Point2}
    s1, s2 = src
    d1, d2 = dest
    @assert s2 >= s1 && d2 >= d1

    if (s1 < d1 && s2 < d1) || (s1 > d2 && s2 > d2) return (1,0), (1,0) end

    r1, r2 = max(s1, d1), min(s2, d2)
    sr1, sr2 = r1-s1+1, r2-s1+1
    dr1, dr2 = r1-d1+1, r2-d1+1
    
    (sr1, sr2), (dr1, dr2)
end

function box_overlap(src::BBox, dest::BBox)
    (xsr1, xsr2), (xdr1, xdr2) = interval_overlap((x1(src), x2(src)), (x1(dest), x2(dest)))
    (ysr1, ysr2), (ydr1, ydr2) = interval_overlap((y1(src), y2(src)), (y1(dest), y2(dest)))

    if ((xsr1, xsr2), (xdr1, xdr2)) == ((1,0), (1,0)) || ((ysr1, ysr2), (ydr1, ydr2)) == ((1,0),(1,0))
        return (bbox(1,1,0,0), bbox(1,1,0,0))
    end
    
    bbox(xsr1, ysr1, xsr2, ysr2), bbox(xdr1, ydr1, xdr2, ydr2)
end

function box_overlap(
    srcsize::Tuple{Int,Int},
    destsize::Tuple{Int,Int},
    topleft::Tuple{Int,Int},
)
    @assert all(srcsize .> 0)
    @assert all(destsize .> 0)

    return box_overlap(bbox(topleft..., (topleft.+srcsize.-1)...), bbox(1, 1, destsize...))
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
