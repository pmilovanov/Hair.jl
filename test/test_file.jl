using Hair;
H = Hair;
using Test, Printf, Images

@testset "File utils" begin

  @testset "Basic read binary data" begin
    tdir = mktempdir()
    for i = 1:3
      bname = @sprintf("%d.png", i)
      write(joinpath(tdir, bname), bname)
      bname = @sprintf("%d-mask.png", i)
      write(joinpath(tdir, bname), bname)
    end

    c = Channel(100)
    @async H.read_images_masks_from_dir(c, tdir)
    @test [String.(x) for x in c] == [
      ("1.png", "1.png", "1-mask.png"),
      ("2.png", "2.png", "2-mask.png"),
      ("3.png", "3.png", "3-mask.png"),
    ]
  end

  @testset "Read images" begin
    tdir = mktempdir()
    N = 3
    A = rand(N, 4, 5, 5)
    imgs = [RGB{N0f8}.(A[i, 1, :, :], A[i, 2, :, :], A[i, 3, :, :]) for i = 1:N]
    masks = [(A[i, 4, :, :] .> 0.9) for i = 1:N]
    for i = 1:N
      save(joinpath(tdir, @sprintf("%d.png", i)), imgs[i])
      save(joinpath(tdir, @sprintf("%d-mask.png", i)), masks[i])
    end

    c_blobs = Channel(100)
    c_imgs = Channel(100)
    @async H.read_images_masks_from_dir(c_blobs, tdir)
    @async H.load_images_masks(c_blobs, c_imgs)
    v = Dict([fname => (img, mask) for (fname, img, mask) in c_imgs])

    @test length(v) == N
    for i = 1:N
      fname = @sprintf("%d.png", i)
      @test v[fname][1] == imgs[i]
      @test (v[fname][2] .> 0.9) == masks[i]
    end

    #a_imgs, a_masks = H.load_data(tdir)

  end

end
