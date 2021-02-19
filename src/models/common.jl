using Flux, CUDA
using Flux: throttle, logitbinarycrossentropy

import ..Upsample, ..SkipUnit
using Parameters: @with_kw
import JSON2

abstract type ModelArgs end

struct AnnotatedModel{M}
  model::M
  metadata::Dict
  function AnnotatedModel(model::M, model_args) where {M}
    d = Dict()
    d[:model_args] = model_args
    new{M}(model, d)
  end
end

(am::AnnotatedModel)(x::AbstractArray) = am.model(x)
model(am::AnnotatedModel) = am.model
metadata(am::AnnotatedModel) = am.metadata
function setmeta!(am::AnnotatedModel, key, value)
  am.metadata[key] = value
  am
end

function savemeta(am::AnnotatedModel, filename::String)
  open(filename, "w") do f
    JSON2.pretty(f, JSON2.write(am.metadata))
    println(f)
  end
end

function conv_block(
  nunits::Int,
  k::Tuple{Int,Int},
  ch::Pair{<:Integer,<:Integer},
  σ = relu;
  pad = (0,0),
  stride = (1, 1),
  kwargs...,
) where {N}
  if pad == (0,0)
    p = (k[1] - 1) ÷ 2
    pad = (p,p)
  end
  chain = [Conv(k, ch, σ; pad = pad, stride = stride, kwargs...), BatchNorm(last(ch))]
  for i = 1:nunits-1
    push!(chain, Conv(k, last(ch) => last(ch), σ; pad = pad, stride = stride, kwargs...))
    σ != selu && push!(chain, BatchNorm(last(ch)))
  end
  Chain(chain...)
end

function upsample_conv(channels::Pair{Int,Int}, σ = relu)
  chain = [Upsample(), Conv((5, 5), channels, σ, pad = SamePad(), stride = (1, 1))]
  σ != selu && push!(chain, BatchNorm(last(channels)))
  Chain(chain...)
end
