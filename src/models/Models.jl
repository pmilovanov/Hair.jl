module Models

using Flux, CUDA
using Flux: throttle, logitbinarycrossentropy

import ..Upsample, ..SkipUnit


function conv_block(
  nunits::Int,
  k::Tuple{Int,Int},
  ch::Pair{<:Integer,<:Integer},
  σ = relu;
  pad = (1, 1),
  stride = (1, 1),
  kwargs...,
) where {N}
  chain = [Conv(k, ch, σ; pad = pad, stride = stride, kwargs...), BatchNorm(last(ch))]
  for i = 1:nunits-1
    push!(chain, Conv(k, last(ch) => last(ch), σ; pad = pad, stride = stride, kwargs...))
    push!(chain, BatchNorm(last(ch)))
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

function build_model_simple(blocksizes::Vector{Int})

  maxpool() = MaxPool((2, 2))

  # transpose conv, upsamples 2x
  upsample_conv(channels::Pair{Int,Int}) =
    Chain(Upsample(), Conv((5, 5), channels, relu, pad = SamePad(), stride = (1, 1)))


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


end
