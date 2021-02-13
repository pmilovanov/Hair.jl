module Models

using Flux, CUDA
using Flux: throttle, logitbinarycrossentropy

import ..Upsample, ..SkipUnit
using Parameters: @with_kw

function conv_block(
  nunits::Int,
  k::Tuple{Int,Int},
  ch::Pair{<:Integer,<:Integer},
  σ = relu;
  pad = SamePad(),
  stride = (1, 1),
  kwargs...,
) where {N}
  chain = [Conv(k, ch, σ; pad = pad, stride = stride, kwargs...), BatchNorm(last(ch))]
  for i = 1:nunits-1
    push!(chain, Conv(k, last(ch) => last(ch), σ; pad = pad, stride = stride, kwargs...))
    if σ != selu
      push!(chain, BatchNorm(last(ch)))
    end
  end
  Chain(chain...)
end

# function build_model(args::TrainArgs = TrainArgs())

#   maxpool() = MaxPool((2, 2))

#   convs3 = Chain( # DebugPrintSize("conv0"),
#                   conv_block(2, (3, 3), 3 => 64), # DebugPrintSize("convs1"),
#                   maxpool(),
#                   conv_block(3, (3, 3), 64 => 128), # DebugPrintSize("convs2"),
#                   maxpool(),
#                   conv_block(3, (3, 3), 128 => 256), # DebugPrintSize("convs3"),
#                   )

#   convs5 = Chain(
#     convs3,
#     maxpool(),
#     conv_block(3, (3, 3), 256 => 512), # DebugPrintSize("convs4"),
#     maxpool(),
#     conv_block(3, (3, 3), 512 => 512), # DebugPrintSize("convs5"),
#   )

#   # transpose conv, upsamples 2x
#   upsample_conv(channels::Pair{Int,Int}) =
#     Chain(Upsample(), Conv((3, 3), channels, relu, pad = SamePad(), stride = (1, 1)))

#   convs5u2 = Chain(
#     convs5,
#     upsample_conv(512 => 256),
#     BatchNorm(256),
#     upsample_conv(256 => 256),
#     BatchNorm(256),
#   )

#   stacked1u2 = Chain(
#     StackChannels(convs5u2, convs3),
#     upsample_conv(512 => 64),
#     BatchNorm(256),
#     upsample_conv(64 => 1),
#   )

#   return stacked1u2

# end

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
  blocksizes::Vector{Int} = [5, 5, 5, 5]
  kernelsizes::Vector{NTuple{2,Int}} = [(7, 7), (7, 7), (7, 7), (7, 7)]
end

maxpool() = MaxPool((2, 2))

function selu_simple(a::SeluSimpleArgs = SeluSimpleArgs())

  convs3 = Chain(# DebugPrintSize("conv0"),
    conv_block(a.blocksizes[1], a.kernelsizes[1], 3 => 16, selu),
    # DebugPrintSize("convs1"),
    maxpool(),
    conv_block(a.blocksizes[2], a.kernelsizes[2], 16 => 24, selu),
    # DebugPrintSize("convs2"),
    maxpool(),
    conv_block(a.blocksizes[3], a.kernelsizes[3], 24 => 32, selu),
    # DebugPrintSize("convs3"),
  )
  convs4u1 = Chain(
    maxpool(),
    conv_block(a.blocksizes[4], a.kernelsizes[4], 32 => 64, selu),
    #   DebugPrintSize("convs4"),
    upsample_conv(64 => 32, selu),
    # BatchNorm(32),
    #    DebugPrintSize("convs4u1"),
  )
  stacked1u2 = Chain(
    SkipUnit(convs3, convs4u1),
    upsample_conv(64 => 16, selu),
    # BatchNorm(16),
    Upsample(),
    Conv((5, 5), 16 => 1, σ, pad = SamePad(), stride = (1, 1)),
    #    DebugPrintSize("stacked1u2"),
  )

  return stacked1u2


end


end
