using Flux, CUDA
using Flux: throttle, logitbinarycrossentropy

import ..Upsample, ..SkipUnit, ..DebugPrintSize
using Parameters: @with_kw
import JSON2

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
    #    DebugPrintSize("convs4u1"),
  )
  stacked1u2 = Chain(
    SkipUnit(convs3, convs4u1),
    upsample_conv(64 => 16),
    Upsample(),
    Conv((5, 5), 16 => 1, σ, pad = SamePad(), stride = (1, 1)),
    #    DebugPrintSize("stacked1u2"),
  )

  return AnnotatedModel(stacked1u2, BasicModelArgs(blocksizes))

end

################################################################################
@with_kw struct SimpleArgs <: ModelArgs
  blocksizes::Vector{Int} = [2, 2, 2, 2, 2]
  kernelsizes::Vector{NTuple{2,Int}} = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
  σ::Function = leakyrelu
end

maxpool() = MaxPool((2, 2))

function simple(a::SimpleArgs = SimpleArgs())

  convs3 = Chain(# # DebugPrintSize("conv0"),
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
    # DebugPrintSize("convs4"),
    upsample_conv(64 => 32, a.σ),
    # BatchNorm(32),
    # DebugPrintSize("convs4u1"),
  )
  stacked1u2 = Chain(
    SkipUnit(convs3, convs4u1),
    upsample_conv(64 => 32, a.σ),
    conv_block(a.blocksizes[5], a.kernelsizes[5], 32=>16, a.σ),
    # BatchNorm(16),
    Upsample(),
    conv_block(a.blocksizes[5], a.kernelsizes[5], 16=>16, a.σ),
    Conv((5, 5), 16 => 1, σ, pad = (2,2), stride = (1, 1)),
    # DebugPrintSize("stacked1u2"),
  )

  return AnnotatedModel(stacked1u2, a)
end
