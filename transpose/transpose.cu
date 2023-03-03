#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define NB_THREADS 256
#define BLOCK_DIM 16

template <typename T>
__global__ void transpose(T *x, T *y) {
  // We pad the shared tile by 1 element to avoid bank conflicts.
  __shared__ T tile[BLOCK_DIM][BLOCK_DIM+1];

  unsigned int width = gridDim.x * BLOCK_DIM;
  unsigned int height = gridDim.y * BLOCK_DIM;

  unsigned int i = blockIdx.y * BLOCK_DIM + threadIdx.y;
  unsigned int j = blockIdx.x * BLOCK_DIM + threadIdx.x;

  tile[threadIdx.y][threadIdx.x] = x[i * width + j];

  __syncthreads();

  i = threadIdx.y + blockIdx.x * BLOCK_DIM;
  j = threadIdx.x + blockIdx.y * BLOCK_DIM;
  
  y[(i * height + j)] = tile[threadIdx.x][threadIdx.y]; 
}

int main(void)
{
  size_t width = BLOCK_DIM * 32;
  size_t height = BLOCK_DIM * 32;
  size_t N = width * height;
  int *x, *y, *dev_x, *dev_y;
  x = (int*)malloc(N*sizeof(int));
  y = (int*)malloc(N*sizeof(int));

  cudaMalloc(&dev_x, N*sizeof(int));
  cudaMalloc(&dev_y, N*sizeof(int));
  
  /* Initialisation de x et y*/
  for (size_t i = 0; i < N; i++) {
    x[i] = i;
  }

  cudaMemcpy(dev_x, x, N*sizeof(int), cudaMemcpyHostToDevice);

  /* TODO : Appel au kernel transpose sur les N éléments */
  dim3 grid(width / BLOCK_DIM, height / BLOCK_DIM);
  dim3 threads(BLOCK_DIM, BLOCK_DIM);
  transpose<<<grid, threads>>>(dev_x, dev_y);

  /* TODO : Copie du résultat dans y*/
  cudaMemcpy(y, dev_y, N*sizeof(int), cudaMemcpyDeviceToHost);

  // Vérification.
  bool error = false;
  for (size_t i = 0; i < height-1; ++i) {
    for (size_t j = 0; j < width-1; ++j) {
      if (x[i*width + j] != y[j*width + i]) {
        printf("Different element at %d-%d : X:%d | Y:%d\n", i, j, x[i*width + j], y[j*width + i]);
        error = true;
      }
    }
  }
  if (!error) printf("Transpose ended successfully\n");

  free(x);
  free(y);
  cudaFree(&dev_x);
  cudaFree(&dev_y);
}