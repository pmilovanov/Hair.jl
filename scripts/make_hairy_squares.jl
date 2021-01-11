using Hair
using ArgParse, Hair, ProgressMeter, JLD2, Images, Distributed


H = Hair


function main()
  @load "/tmp/hairs.jld2" hairs
  println("Loaded hairs")

  H.make_hairy_squares(
    hairs,
    "/home/pmilovanov/data/the_met/temp/oil_sample_100",
    "/home/pmilovanov/data/hair/hairy/exp/0107-01",
    H.MakeHairySquaresOptions(max_hairs_per_output = 10),
  )
  println("OK")
end

main()
