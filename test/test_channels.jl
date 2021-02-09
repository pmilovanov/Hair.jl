using Test
using Base.Threads: @spawn

import Hair
using Hair: @spawnlog

H = Hair

tracker = H.StatsTracker()
@show tracker

tch = H.TrackingChannel("ch1", Channel{Int}(100), tracker)

@spawnlog for i=1:100000
  take!(tch)
  sleep(rand()*0.045)
end


@spawnlog begin
  for i = 1:100000
    put!(tch, i)
    sleep(rand()*0.05)
  end
  close(tch)
end


sleep(5.0)

stats = @time H.snapshot(tracker)


for k in sort(collect(keys(stats)))
  println("----------- $k -------------")
  show(stats[k])
end
