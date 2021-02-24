FROM gcr.io/deeplearning-platform-release/base-cu110

WORKDIR /opt
RUN curl https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.0-rc1-linux-x86_64.tar.gz | tar xzv

RUN ln -s julia-1.6.0-rc1 julia
RUN ln -s /opt/julia/bin/julia /usr/local/bin/julia

WORKDIR /sysimgs

WORKDIR /app
COPY *.toml /app/

RUN julia -O2 -t8 --project=. -e "import Pkg; Pkg.precompile()"

#RUN mkdir build
COPY build /app/build/

RUN julia -i --project=. -O2 -t8 -e "using Flux"
RUN julia --project=. -O2 -t8 build/sysimage.jl -o /sysimgs/custom.so

COPY . /app/

ENTRYPOINT ["/app/build/entrypoint.sh"]

RUN julia -i -O2  --project=/app -J /sysimgs/custom.so -e "using Flux; using CUDA"

RUN nvidia-smi
