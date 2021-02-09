using Lazy: @forward
using Base.Threads: @spawn

import Base.isbuffered,
  Base.check_channel_state,
  Base.close,
  Base.isopen,
  Base.isready,
  Base.n_avail,
  Base.isempty,
  Base.lock,
  Base.unlock,
  Base.trylock,
  Base.wait,
  Base.put!,
  Base.fetch,
  Base.take!,
  Base.eltype,
  Base.bind,
  Base.show,
  Base.iterate



import OnlineStats
OS = OnlineStats

abstract type StatsTrackerMessage end

struct Measurement{T<:Number} <: StatsTrackerMessage
  key::String
  value::T
end

struct SnapshotRequest <: StatsTrackerMessage
  outputch::Channel{Dict{String,OS.Series}}
end

struct StatsTracker
  _reportch::Channel{StatsTrackerMessage}


  function StatsTracker(chsize = 10000, quantiles = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99])
    t = new(Channel{StatsTrackerMessage}(chsize))
    @spawnlog _stats_tracker_processing_task(t, quantiles)
    t
  end

end


function _process!(message::Measurement, stats::Dict{String,OS.Series}, series_initializer)
  if !haskey(stats, message.key)
    stats[message.key] = series_initializer()
  end
  fit!(stats[message.key], message.value)
  nothing  # type stability
end

function _process!(message::SnapshotRequest, stats::Dict{String,OS.Series}, series_initializer)
  put!(message.outputch, deepcopy(stats))
  nothing  # type stability
end


function _stats_tracker_processing_task(t::StatsTracker, quantiles::Vector{<:Real})
  stats = Dict{String,OS.Series}()

  series_initializer() =
    OS.Series(OS.Extrema(), OS.Mean(), OS.Variance(), [OS.P2Quantile(x) for x in quantiles]...)

  for message in t._reportch
    _process!(message, stats, series_initializer)
  end
end



report!(t::StatsTracker, key::String, value::Number) = put!(t._reportch, Measurement(key, value))

function snapshot(t::StatsTracker)
  ch = Channel{Dict{String,OS.Series}}(1)  # unbuffered
  @info "gonna put snapshot request"
  @info n_avail(t._reportch)
  put!(t._reportch, SnapshotRequest(ch))
  take!(ch)
end

struct TrackingChannel{T} <: AbstractChannel{T}
  id::String
  ch::Channel{T}
  tracker::StatsTracker
end

@forward TrackingChannel.ch (
  Base.isbuffered,
  Base.check_channel_state,
  Base.close,
  Base.isopen,
  Base.isready,
  Base.n_avail,
  Base.isempty,
  Base.lock,
  Base.unlock,
  Base.trylock,
  Base.wait,
)

Base.eltype(::Type{TrackingChannel{T}}) where {T} = T
Base.bind(c::TrackingChannel, task::Task) = bind(c.chan, task)

Base.show(io::IO, c::TrackingChannel) = print(io, typeof(c), "[id=$(c.id), chan=$(c.ch)]")

report!(c::TrackingChannel, name::String, value::Number) =
  report!(c.tracker, "$(c.id)__$(name)", value)

function Base.put!(c::TrackingChannel{T}, v) where {T}
  n = n_avail(c.ch)
  @assert isa(n, Int)
  t = @elapsed put!(c.ch, v)
  report!(c, "put-time", t)
  report!(c, "put-length", n)
end

function Base.fetch(c::TrackingChannel)
  n = n_avail(c.ch)
  t = @elapsed result = fetch(c.ch)
  report!(c, "fetch-time", t)
  report!(c, "fetch-length", n)
  result
end

function Base.take!(c::TrackingChannel)
  n = n_avail(c.ch)
  t = @elapsed result = take!(c.ch)
  report!(c, "take-time", t)
  report!(c, "take-length", n)
  result
end

function Base.iterate(c::TrackingChannel, state = nothing)
  try
    return (take!(c), nothing)
  catch e
    if isa(e, InvalidStateException) && e.state === :closed
      return nothing
    else
      rethrow()
    end
  end
end
