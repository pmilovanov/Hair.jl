using Flux
using Flux.Data: DataLoader


################################################################################
# GPU data loader
################################################################################

struct GPUDataLoader{D}
  inner::DataLoader{D}
end

function Base.iterate(d::GPUDataLoader, i = 0)
  it = Base.iterate(d.inner, i)
  if it == nothing
    return nothing
  else
    data, nexti = it
    return (gpu(data), nexti)
  end
end
Base.length(d::GPUDataLoader) = Base.length(d.inner)

Base.eltype(::GPUDataLoader{D}) where {D} = D
