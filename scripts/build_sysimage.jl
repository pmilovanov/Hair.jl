EXCLUDE_PACKAGES = ["ImageView"]

#
#
#

import Pkg

cd(@__DIR__)
cd("..")
Pkg.activate(".")

Pkg.precompile()

using PackageCompiler
using ArgParse


s = ArgParseSettings()
@add_arg_table s begin
  "--output", "-o"
    arg_type=String
    default = expanduser("~/.julia/sysimgs/custom.so")
end
args = parse_args(ARGS, s)


@info "Running tests"
@info "================================================================================"
include("test/runtests.jl")

symbols = [Symbol(x) for x in keys(Pkg.project().dependencies) if !(x in EXCLUDE_PACKAGES)]

@info "Creating system image"
@info "================================================================================"
create_sysimage(
  symbols,
  sysimage_path = args.output,
)
