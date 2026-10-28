# ==============================================================================
# GPU-Accelerated 2D Heat Diffusion Solver: Dockerfile
# Supports Jacobi (JOR), Red-Black SOR, and Geometric Multigrid (GMG) with CUDA Graphs
# Architectures: Turing (T4)  
# ==============================================================================

FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

LABEL maintainer="Peteti Charantej (G25AIT2026), Aryaveer Rathi (G25AIT2021), Sandeep (G25AIT2036)"
LABEL description="CUDA C++ 2D Heat Diffusion & Multigrid Solver with JupyterLab environment"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONUNBUFFERED=1

# Install the host C++ toolchain used by nvcc, and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install scientific computing & Jupyter environment
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir \
    numpy \
    matplotlib \
    pandas \
    jupyterlab \
    nbconvert

WORKDIR /workspace

# Copy source code, build configuration and notebook into the container
COPY heat2d.cu /workspace/
COPY gmg.cu /workspace/
COPY Makefile /workspace/
COPY G25ait2026_final.ipynb /workspace/
COPY README.md /workspace/

# Build both CUDA solvers with multi-architecture support
# (sm_75: T4, sm_80: A100, sm_86: RTX 30xx, sm_89: RTX 40xx, sm_90: H100, compute_75: forward PTX)
RUN nvcc -O3 -std=c++17 -lineinfo \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_90,code=sm_90 \
    -gencode arch=compute_75,code=compute_75 \
    heat2d.cu -o heat2d && \
    nvcc -O3 -std=c++17 -lineinfo \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_90,code=sm_90 \
    -gencode arch=compute_75,code=compute_75 \
    gmg.cu -o gmg

# Expose port for JupyterLab
EXPOSE 8888

# Default execution: Run GMG solver benchmark (N=513, eps=1e-4)
# Pass 'jupyter' or bash as container command to launch JupyterLab or interactive shell
CMD ["./gmg", "-n", "513", "-e", "1e-4"]
