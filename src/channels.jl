using Lazy: @forward
import OnlineStats
OS = OnlineStats

_online_stats_series(quantiles::Vector{<:Real}) =
  OS.Series(OS.Extrema(), OS.Mean(), OS.Variance(), [OS.P2Quantile(x) for x in quantiles]...)


struct StatsTracker
  _reportch::Channel{Tuple{String,Number}}
  _quantiles::Vector{<:Real}
  _stats::Dict{String,OS.OnlineStat}

  # Lock used so that we don't try to modify online stats when a snapshot is being taken.
  # Most of the time it's held inside the iterations in the loop of the measurement-processing task.
  _stats_lock::ReentrantLock

  function StatsTracker(chsize = 10000, quantiles = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99])
    t = new(
      Channel{Tuple{String,Number}}(chsize),
      quantiles,
      Dict{String,OS.OnlineStat}(),
      ReentrantLock(),
    )

    @spawn for (k, v) in t._reportch
      lock(t._stats_lock) do
        if !haskey(t._stats, k)
          t._stats[k] = _online_stats_series(t._quantiles)
        end
        fit!(t._stats[k], v)
      end
    end
  end
end

report!(t::StatsTracker, key::String, value::Number) = put!(t._reportch, (key, value))

snapshot(t::StatsTracker) =
  lock(t._stats_lock) do
    return deepcopy(t._stats)
  end


struct LoggingChannel{T} <: AbstractChannel{T}
  id::String
  ch::Channel{T}
  tracker::StatsTracker
end


@forward LoggingChannel.chan (
  isbuffered,
  check_channel_state,
  close,
  isopen,
  isready,
  n_avail,
  isempty,
  lock,
  unlock,
  trylock,
  wait,
)


eltype(::Type{LoggingChannel{T}}) where {T} = T

bind(c::LoggingChannel, task::Task) = bind(c.chan, task)


show(io::IO, c::LoggingChannel) = print(io, typeof(c), "[id=$(id), chan=$(chan)]")



report!(c::LoggingChannel, name::String, value::Number) =
  report!(c.tracker, "$(c.id)__$(name)", value)

function put!(c::LoggingChannel{T}, v) where {T}
  n = n_avail(c.ch)
  t = @elapsed put!(c.ch, v)
  _report!(c, "put-time", t)
  _report!(c, "put-length", t)
end

function fetch(c::Channel)
  n = n_avail(c.ch)
  t = @elapsed result = fetch(c.ch)
  _report!(c, "fetch-time", t)
  _report!(c, "fetch-length", n)
  result
end

function take!(c::Channel)
  n = n_avail(c.ch)
  t = @elapsed result = take!(c.ch)
  _report!(c, "take-time", t)
  _report!(c, "take-length", n)
  result
end
