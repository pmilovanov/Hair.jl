using Hair
using ArgParse, Hair, ProgressMeter, JLD2, Images, Distributed


H = Hair


function main()
  @load expanduser("~/data/hair/hair_db.jld2") hairs
  println("Loaded hairs")

  indir = "/home/pmilovanov/data/the_met/temp/oil_sample_100"
  outdir = "/home/pmilovanov/data/hair/hairy/exp/0113-01"

  if !isdir(outdir)
    mkdir(outdir)
  end

  H.make_hairy_squares(
    hairs,
    indir,
    outdir,
    H.MakeHairySquaresOptions(samples_per_pic = 2, max_hairs_per_output = 10),
  )
  println("OK")
end

main()
