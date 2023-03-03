CC=gcc

CUDA_HOME=/usr/local/cuda
CUDASDK=/usr/local/cuda/samples
CUDANVIDIA=/usr/lib64/nvidia
NVCC=$(CUDA_HOME)/bin/nvcc

NVCFLAGS= -I $(CUDA_HOME)/include -I$(CUDASDK)/common/inc

CFLAGS= -L $(CUDA_HOME)/lib64 -L $(CUDANVIDIA)

saxpygpu3: saxpy.o
	$(NVCC) $(CFLAGS)  -o $@ $^ 

clean:
	rm -f *.o *.linkinfo saxpygpu3

%.o: %.cu
	$(NVCC) $(NVCFLAGS) -c -o $@ $<