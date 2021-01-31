struct LoggingChannel{T} <: AbstractChannel{T}
  _channel::Channel{T}
  _logto::Channel
end
