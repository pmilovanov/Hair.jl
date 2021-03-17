using Flux, CUDA
using Flux: throttle, logitbinarycrossentropy

import ..Upsample, ..SkipUnit, ..gcscopy, ..isgcs, ..gsispath
using Parameters: @with_kw
import JSON2
using Printf
using BSON: @save

abstract type ModelArgs end

struct BasicModelArgs{T} <: ModelArgs
    args::T
end

@with_kw mutable struct AnnotatedModel
    model
    metadata::Dict
    epoch::Int = 0
end

function AnnotatedModel(model, model_args::ModelArgs)  
    d = Dict()
    d[:model_args] = model_args
    AnnotatedModel(model=model, metadata=d)
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



function savemodel(am::AnnotatedModel, dir::String)
    localdir = isgcs(dir) ? mktempdir() : dir

    modelbasename = @sprintf("epoch_%03d.bson", am.epoch)
    localmodelfname = joinpath(localdir, modelbasename)
    actualmodelfname = joinpath(dir, modelbasename)
    localmetafname = joinpath(localdir, @sprintf("epoch_%03d.json", am.epoch))
    gsispath(actualmodelfname) && throw(InvalidStateException("File $localmodelfname already exists", :FILE_ALREADY_EXISTS))
  
    model = AnnotatedModel(model=cpu(am.model), metadata=am.metadata, epoch=am.epoch)

    @save localmodelfname model
    Models.savemeta(am, localmetafname)
  
    @info "Saved model locally to $localmodelfname"

    if isgcs(dir)
        gcscopy(localmodelfname, joinpath(dir, ""))
        gcscopy(localmetafname, joinpath(dir, ""))
        @info "Saved model on gcs to $(joinpath(dir, basename(localmodelfname)))"
    end
end
  

function conv_block(
  nunits::Int,
  k::Tuple{Int,Int},
  ch::Pair{<:Integer,<:Integer},
  σ=relu;
  pad=(0, 0),
  stride=(1, 1),
  kwargs...,
) where {N}
    if pad == (0, 0)
        p = (k[1] - 1) ÷ 2
        pad = (p, p)
    end
    chain = [Conv(k, ch, σ; pad=pad, stride=stride, kwargs...), BatchNorm(last(ch))]
    for i = 1:nunits - 1
        push!(chain, Conv(k, last(ch) => last(ch), σ; pad=pad, stride=stride, kwargs...))
        σ != selu && push!(chain, BatchNorm(last(ch)))
    end
    Chain(chain...)
end

function upsample_conv(channels::Pair{Int,Int}, σ=relu)
    chain = [Upsample(), Conv((5, 5), channels, σ, pad=SamePad(), stride=(1, 1))]
    σ != selu && push!(chain, BatchNorm(last(channels)))
    Chain(chain...)
end
