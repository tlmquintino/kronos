// inc_array.cu

#if USE_DOUBLE
typedef real_t real_t;
#else
typedef float  real_t;
#endif

#include <stdio.h>
#include <assert.h>
#include <cuda.h>

void incrementArrayOnHost(real_t *a, int N)
{
  int i;
  for (i=0; i < N; i++) a[i] = a[i]+1.f;
}

__global__ void incrementArrayOnDevice(real_t *a, int N)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx<N) a[idx] = a[idx]+1.f;
}

int main(void)
{
  real_t *a_h, *b_h;           // pointers to host memory
  real_t *a_d;                 // pointer to device memory
  int i, N = 32*1024;
  size_t size = N*sizeof(real_t);
  
  // allocate arrays on host
  a_h = (real_t *)malloc(size);
  b_h = (real_t *)malloc(size);
  
  // allocate array on device 
  cudaMalloc((void **) &a_d, size);
  
  // initialization of host data
  for (i=0; i<N; i++) a_h[i] = (real_t)i;
  // copy data from host to device
  cudaMemcpy(a_d, a_h, sizeof(real_t)*N, cudaMemcpyHostToDevice);
  
  // do calculation on host
  incrementArrayOnHost(a_h, N);
  
  // do calculation on device:
  
  // Part 1 of 2. Compute execution configuration
  int blockSize = 4;
  int nBlocks = N/blockSize + (N%blockSize == 0?0:1);
  
  // Part 2 of 2. Call incrementArrayOnDevice kernel 
  incrementArrayOnDevice <<< nBlocks, blockSize >>> (a_d, N);
  
  // Retrieve result from device and store in b_h
  cudaMemcpy(b_h, a_d, sizeof(real_t)*N, cudaMemcpyDeviceToHost);
  
  // check results
  for (i=0; i<N; i++) assert(a_h[i] == b_h[i]);
  
  // print
  /* for (i=0; i<N; i++) printf( "a_h[i] = %f\n", a_h[i]); */
  
  // cleanup
  free(a_h); 
  free(b_h); 
  cudaFree(a_d); 
  }
