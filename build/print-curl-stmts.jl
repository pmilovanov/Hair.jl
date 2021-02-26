import Pkg, Artifacts, CUDA

toml = Artifacts.find_artifacts_toml(pathof(CUDA))

artifacts = Artifacts.load_artifacts_toml(toml)

for artname in ["CUDA110", "CUDNN_CUDA110", "CUTENSOR_CUDA110", "CUDA111", "CUDNN_CUDA111", "CUTENSOR_CUDA111"]

  a = artifacts[artname]

  a = [x for x in a if x["arch"] == "x86_64" && x["os"] == "linux"][1]

  url = a["download"][1]["url"]
  sha = a["git-tree-sha1"]

  println("curl -L $(url) -o $(sha).tar.gz")

end

