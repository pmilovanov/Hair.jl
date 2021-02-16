module Models

using Flux, CUDA
using Flux: throttle, logitbinarycrossentropy

import ..Upsample, ..SkipUnit
using Parameters: @with_kw


abstract type ModelArgs end

struct ModelWArgs{M,A<:ModelArgs}
  model::M
  args::A
end

(mwa::ModelWArgs)(x::AbstractArray) = mwa.model(x)
model(mwa::ModelWArgs) = mwa.model
args(mwa::ModelWArgs) = mwa.args

function conv_block(
  nunits::Int,
  k::Tuple{Int,Int},
  ch::Pair{<:Integer,<:Integer},
  σ = relu;
  pad = (1,1),
  stride = (1, 1),
  kwargs...,
) where {N}
  chain = [Conv(k, ch, σ; pad = pad, stride = stride, kwargs...), BatchNorm(last(ch))]
  for i = 1:nunits-1
    push!(chain, Conv(k, last(ch) => last(ch), σ; pad = pad, stride = stride, kwargs...))
    if σ != selu
      @info "hi ho let's go"
      push!(chain, BatchNorm(last(ch)))
    end
  end
  Chain(chain...)
end

upsample_conv(channels::Pair{Int,Int}, σ = relu) =
  Chain(Upsample(), Conv((5, 5), channels, σ, pad = SamePad(), stride = (1, 1)))

function build_model_simple(blocksizes::Vector{Int})

  maxpool() = MaxPool((2, 2))

  # transpose conv, upsamples 2x

  convs3 = Chain(#  DebugPrintSize("conv0"),
    conv_block(blocksizes[1], (3, 3), 3 => 16), # DebugPrintSize("convs1"),
    maxpool(),
    conv_block(blocksizes[2], (3, 3), 16 => 24), # DebugPrintSize("convs2"),
    maxpool(),
    conv_block(blocksizes[3], (3, 3), 24 => 32),  #DebugPrintSize("convs3"),
  )
  convs4u1 = Chain(
    maxpool(),
    conv_block(blocksizes[4], (3, 3), 32 => 64),
    #   DebugPrintSize("convs4"),
    upsample_conv(64 => 32),
    BatchNorm(32),
    #    DebugPrintSize("convs4u1"),
  )
  stacked1u2 = Chain(
    SkipUnit(convs3, convs4u1),
    upsample_conv(64 => 16),
    BatchNorm(16),
    Upsample(),
    Conv((5, 5), 16 => 1, σ, pad = SamePad(), stride = (1, 1)),
    #    DebugPrintSize("stacked1u2"),
  )

  return stacked1u2

end

################################################################################
@with_kw struct SeluSimpleArgs
  blocksizes::Vector{Int} = [2, 2, 2, 2, 2]
  kernelsizes::Vector{NTuple{2,Int}} = [(3, 3), (3, 3), (3, 3), (3, 3), (3,3)]
  σ::Function = relu
end

maxpool() = MaxPool((2, 2))

function selu_simple(a::SeluSimpleArgs = SeluSimpleArgs())

  convs3 = Chain(# DebugPrintSize("conv0"),
    conv_block(a.blocksizes[1], a.kernelsizes[1], 3 => 16, a.σ),
    # DebugPrintSize("convs1"),
    maxpool(),
    conv_block(a.blocksizes[2], a.kernelsizes[2], 16 => 24, a.σ),
    # DebugPrintSize("convs2"),
    maxpool(),
    conv_block(a.blocksizes[3], a.kernelsizes[3], 24 => 32, a.σ),
    # DebugPrintSize("convs3"),
  )
  convs4u1 = Chain(
    maxpool(),
    conv_block(a.blocksizes[4], a.kernelsizes[4], 32 => 64, a.σ),
    #   DebugPrintSize("convs4"),
    upsample_conv(64 => 32, a.σ),
    # BatchNorm(32),
    #    DebugPrintSize("convs4u1"),
  )
  stacked1u2 = Chain(
    # SkipUnit(convs3, convs4u1),
    convs3,
    convs4u1,
    upsample_conv(32 => 16, a.σ),
#    conv_block(a.blocksizes[5], a.kernelsizes[5], 16=>16, a.σ)  
    # BatchNorm(16),
    Upsample(),
 #   conv_block(a.blocksizes[5], a.kernelsizes[5], 16=>16, a.σ)
    Conv((5, 5), 16 => 1, σ, pad = SamePad(), stride = (1, 1)),
    #    DebugPrintSize("stacked1u2"),
  )

  return stacked1u2


end


end
