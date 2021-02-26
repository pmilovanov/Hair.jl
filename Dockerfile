FROM gcr.io/dada-science/gpu-julia-1.6.0-rc1

#FROM gcr.io/deeplearning-platform-release/base-cu110
#FROM nvidia/cuda:11.2.1-cudnn8-runtime-ubuntu20.04

WORKDIR /app
COPY *.toml /app/

RUN julia -O2 -t8 --project=. -e "import Pkg; Pkg.precompile()"

#ADD build /app/build/


COPY . /app/

ENTRYPOINT ["/app/build/entrypoint.sh"]


