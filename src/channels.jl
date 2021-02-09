using Lazy: @forward


struct LoggingChannel{T} <: AbstractChannel{T}
  id::String
  chan::Channel{T}
  reportchan::Channel{Tuple{String, Number}}
end

@forward LoggingChannel.chan isbuffered, check_channel_state, close, isopen, isready, n_avail, isempty, lock, unlock, trylock, wait

eltype(::Type{LoggingChannel{T}}) where {T} = T

bind(c::LoggingChannel, task::Task) = bind(c.chan, task)



show(io::IO, c::LoggingChannel) = print(io, typeof(c), "[id=$(id), chan=$(chan)]")



function _report!(c::LoggingChannel, name::String, value::Number)
  label = "$(c.id)__$(name)"
  put!(c.reportchan, (label, value))
end

function put!(c::LoggingChannel{T}, v) where T
  t = @elapsed put!(c.chan, v)
  _report!(c, "put-time", t)
end

function fetch(c::Channel)
  n = n_avail(c.chan)
  t = @elapsed result = fetch(c.chan)
  _report!(c, "fetch-time", t)
  _report!(c, "fetch-length", n)
  result
end

function take!(c::Channel)
  n = n_avail(c.chan)
  t = @elapsed result = take!(c.chan)
  _report!(c, "take-time", t)
  _report!(c, "take-length", n)
  result 
end


