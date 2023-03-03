#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

/********************** kernel **************************/
__global__ void saxpy(int n, float a, float *x, float *y)
{
  /* TODO : Calcul de l'indice i*/
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

/********************** main **************************/
int main(void)
{
  int N = 1<<20;
  float *x, *y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  /* TODO : Allocation de l'espace pour gpu_x et gpu_y qui 
     vont recevoir x et y sur le GPU*/
  float *dev_x, *dev_y;
  cudaMalloc((void**)&dev_x, N*sizeof(float));
  cudaMalloc((void**)&dev_y, N*sizeof(float));
  
  /* Initialisation de x et y*/
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
  
  /* TODO : Copie de x et y sur le GPU dans gpu_x et gpu_y respectivement*/
  cudaMemcpy(dev_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
  /* TODO : Appel au kernel saxpy sur les N éléments avec a=2.0f */
  saxpy<<<(N+255)/256, 256>>>(N, 2.0, dev_x, dev_y);
  cudaDeviceSynchronize();
  
  /* TODO : Copie du résultat dans y*/
  cudaMemcpy(y, dev_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    {
      if(y[i] != 4.0f)
	      printf("not equal %d %f %f\n", i, y[i], x[i]);
      maxError = max(maxError, abs(y[i]-4.0f));
    }
  printf("Max error: %f\n", maxError);

  /* TODO : Libération de la mémoire sur le GPU*/
  free(x);
  free(y);
  cudaFree(&dev_x);
  cudaFree(&dev_y);
}