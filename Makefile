NVCC := nvcc
CXX_FLAGS := -O3 -std=c++17 -lineinfo

# Target architectures:
# sm_75: Tesla T4, RTX 20-series
# sm_80: A100
# sm_86: RTX 30-series
# sm_89: RTX 40-series (Ada Lovelace)
# sm_90: H100 (Hopper)
# compute_75: Forward PTX JIT for newer/Blackwell architectures
GENCODE_FLAGS := \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_90,code=sm_90 \
    -gencode arch=compute_75,code=compute_75

IMAGE_NAME := gpu-heat-diffusion
TAG := latest

.PHONY: all clean run-gmg run-heat2d docker-build docker-run docker-jupyter

all: heat2d gmg

heat2d: heat2d.cu
	$(NVCC) $(CXX_FLAGS) $(GENCODE_FLAGS) $< -o $@

gmg: gmg.cu
	$(NVCC) $(CXX_FLAGS) $(GENCODE_FLAGS) $< -o $@

run-heat2d: heat2d
	./heat2d -n 256 -e 1e-4 -m all -v both -p double

run-gmg: gmg
	./gmg -n 513 -e 1e-4 -c V -p double

docker-build:
	docker build -t $(IMAGE_NAME):$(TAG) .

docker-run:
	docker run --gpus all --rm -it $(IMAGE_NAME):$(TAG)

docker-jupyter:
	docker run --gpus all --rm -it -p 8888:8888 $(IMAGE_NAME):$(TAG) \
	    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

clean:
	rm -f heat2d gmg *.o *.csv *.bin
