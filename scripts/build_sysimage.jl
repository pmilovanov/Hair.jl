using Hair

using ArgParse
using Base.Threads
using Distributed
using Flux
using CUDA
using ImageMagick
using ImageView
using Images
using Interpolations
using JLD2
using LinearAlgebra
using MosaicViews
using OffsetArrays
using Parameters
using Printf
using ProgressMeter
using Random
using Statistics
using StatsBase
using Test
using TestImages
using Revise
using JuliaFormatter




curdir = pwd()
cd(@__DIR__)
include("../test/runtests.jl")


cd(expanduser("~/.julia/sysimgs"))

using PackageCompiler

@info "Creating system image"
try
  create_sysimage(
    [
      :ArgParse,
      :Distributed,
      :Flux,
      :CUDA,
      :ImageMagick,
      :ImageView,
      :Images,
      :Interpolations,
      :JLD2,
      :LinearAlgebra,
      :MosaicViews,
      :OffsetArrays,
      :Parameters,
      :Printf,
      :ProgressMeter,
      :Random,
      :Statistics,
      :StatsBase,
      :Test,
      :TestImages,
      :Revise,
      :JuliaFormatter,
      #:Hair,
    ],
    sysimage_path = "custom1.so",
  )
catch e
  @error e
finally
  cd(curdir)
end
