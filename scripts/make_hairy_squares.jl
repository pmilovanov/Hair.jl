using Hair
using ArgParse, Hair, ProgressMeter, JLD2, Images, Distributed


H = Hair


function main()
  @load expanduser("~/data/hair/hair_db.jld2") hairs
  println("Loaded hairs")

  indir = "/home/pmilovanov/data/the_met/oil-zeros"
  outdir = "/home/pmilovanov/data/hair/hairy/exp/2k_256_0222"

  if !isdir(outdir)
    mkdir(outdir)
  end

  H.make_hairy_squares(
    hairs,
    indir,
    outdir,
    H.MakeHairySquaresOptions(samples_per_pic = 1, max_hairs_per_output = 20, square_size = 256),
  )
  println("OK")
end

main()
