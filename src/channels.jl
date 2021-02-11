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

####################################################################################################
"""
Tracks statistics of asynchronously reported observations.

Uses OnlineStats.jl package to keep track of stats in a lightweight way and
does not store the actual observations.

- `report!(tracker, key::String, value::Number)`: 
     records a measurement under a string key and updates its statistics.

- `snapshot(tracker)`: gets a copy of the current stats, a dictionary of String->OnlineStats.Series

Constructor takes buffer length for the channel (`chsize`) and the list of quantiles to track.
Additional stats tracked are mean, variance, and extrema.
"""
struct StatsTracker
  #=
  A key part of the tracker is the task implemented by `_stats_tracker_processing_task` below,
  which runs in a separate thread.
  It reads in an infinite loop from the `_reportch`, updates stats and provides snapshots of
  the stats.

  The stats data lives in the local function scope of the task. That way it can only be directly
  read and written by the task itself, synchronization is implemented with the `_reportch` channel
  and we don't have to have an extra lock for taking the snapshot of the stats.
  =# 
  
  _reportch::Channel{StatsTrackerMessage}
  
  function StatsTracker(chsize = 10000, quantiles = [0.5, 0.95, 0.99])
    t = new(Channel{StatsTrackerMessage}(chsize))
    @spawn _stats_tracker_processing_task(t, quantiles)
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
@inline function report!(::Nothing, key::String, value::Number); end

"Get a copy of the current stats of the tracker"
function snapshot(t::StatsTracker)::Dict{String, OS.Series}
  ch = Channel{Dict{String,OS.Series}}(1)
  put!(t._reportch, SnapshotRequest(ch))
  take!(ch)
end

####################################################################################################
"""
Wraps an internal channel and tracks performance statistics for `take!`, `put!`, `fetch` calls.

Performance stats are blocking time and length of data buffer and they are reported to the StatsTracker
provided on construction. E.g if we have `ch1 = TrackingChannel("ch1", Channel(100), tracker)`,
calls to `take!(ch1)` will report measurements named `"ch1__take-time"` and `"ch1__take-length"` to the
`tracker`.

`tracker` can also be set to `nothing` and then nothing is reported anywhere.
"""
struct TrackingChannel{T} <: AbstractChannel{T}
  id::String
  ch::Channel{T}
  tracker::Union{StatsTracker, Nothing}
end

TrackingChannel(id::String, ch::Channel{T}) where T = TrackingChannel(id, ch, nothing)

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

@inline report!(c::TrackingChannel, name::String, value::Number) =
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
