#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_DIM 16

template <typename T>
__global__ void matrixMultiplicationKernel(T* A, T* B, T* C, size_t A_width, size_t A_height, size_t B_width, size_t B_height) {
  __shared__ T A_tile[BLOCK_DIM][BLOCK_DIM];
  __shared__ T B_tile[BLOCK_DIM][BLOCK_DIM];

  T C_element = 0;
    
  // Fill corresponding tiles and accumulate into the new element for C.
  for (size_t tile_id = 0; tile_id < ceilf(A_width / BLOCK_DIM); ++tile_id) {
      size_t i = blockIdx.y * blockDim.y + threadIdx.y;
      size_t j = tile_id * blockDim.x + threadIdx.x;
      A_tile[threadIdx.y][threadIdx.x] = ((i<A_width) && (j<A_height)) ? A[i * A_width + j] : 0;

      i = tile_id * blockDim.y + threadIdx.y;
      j = blockIdx.x * blockDim.x + threadIdx.x;
      B_tile[threadIdx.y][threadIdx.x] = ((i<B_width) && (j<B_height)) ? B[i * B_width + j] : 0;

      __syncthreads();

      for (size_t id = 0; id < BLOCK_DIM; ++id) 
          C_element += A_tile[threadIdx.y][id] * B_tile[id][threadIdx.x];
  
      __syncthreads();
  }

  C[(blockIdx.y * blockDim.y + threadIdx.y) * B_width + (blockIdx.x * blockDim.x + threadIdx.x)] = C_element;
}

int main(int argc, char** argv) {
  size_t A_width = BLOCK_DIM * 16;
  size_t A_height = BLOCK_DIM * 16;
  size_t B_width = BLOCK_DIM * 16;
  size_t B_height = BLOCK_DIM * 16;

  float *A, *B, *C, *dev_A, *dev_B, *dev_C;
  A = (float*)malloc(A_width * A_height * sizeof(float));
  B = (float*)malloc(B_width * B_height * sizeof(float)); 
  C = (float*)malloc(B_width * A_height * sizeof(float)); 

  cudaMalloc(&dev_A, A_width * A_height * sizeof(float)); 
  cudaMalloc(&dev_B, B_width * B_height * sizeof(float));
  cudaMalloc(&dev_C, B_width * A_height * sizeof(float));

  std::random_device rand;
  std::default_random_engine engine(rand());
  std::uniform_real_distribution<float> uni_dist(-1,1);

  // Initialize matrixes with random elements.
  for (size_t i = 0; i < A_width * A_height; i++) {
    A[i] = uni_dist(engine);
  }
  for (size_t i = 0; i < B_width * B_height; i++) {
    B[i] = uni_dist(engine);
  }

  cudaMemcpy(dev_A, A, A_width * A_height * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B, B, B_width * B_height * sizeof(float), cudaMemcpyHostToDevice);  

  dim3 grid(B_width / BLOCK_DIM, A_height / BLOCK_DIM);
  dim3 threads(BLOCK_DIM, BLOCK_DIM);
  matrixMultiplicationKernel<<<grid, threads>>>(dev_A, dev_B, dev_C, A_width, A_height, B_width, B_height);

  cudaMemcpy(C, dev_C, B_width * A_height * sizeof(float), cudaMemcpyDeviceToHost);

  // Verification.
  for (size_t i = 0; i < B_width; ++i) {
    for (size_t j = 0; j < A_height; ++j) {
      float sum = 0;
      for (size_t k = 0; k < A_width; ++k) {
        sum += A[i * A_width + k] * B[k * B_width + j];
      }
      if (std::abs(sum - C[i * B_width + j]) > std::pow(10, -5)) printf("Differed by more than 10e-5 at %d-%d : %f | %f\n", i, j, sum, C[i * B_width + j]);
    }
  }

  free(A);
  free(B);
  free(C);

  cudaFree(&dev_A);
  cudaFree(&dev_B);
  cudaFree(&dev_C);
}