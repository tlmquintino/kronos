#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// #include <cuda.h>
// #include <cuda_runtime.h>

#include "matrix_sizes.h"
#include "matrix_mult.h"

#if 0 // version 1
// CUDA Kernel
__global__ void
cudakernel_matrix_mul( real_t* C, real_t* A, real_t* B, int wA, int wB)
{

   // 2D Thread ID
   int tx = threadIdx.x;
   int ty = threadIdx.y;

   // value stores the element that is
   // computed by the thread
   real_t value = 0;
   for (int i = 0; i < wA; ++i)
   {
      real_t elementA = A[ty * wA + i];
      real_t elementB = B[i * wB + tx];
      value += elementA * elementB;
   }

   // Write the matrix to device memory each
   // thread writes one element
   C[ty * wA + tx] = value;
}
#endif

#if 1 // version 2
// CUDA Kernel
// Multiply two matrices A * B = C
__global__ void
cudakernel_matrix_mul( real_t* C, real_t* A, real_t* B, int wA, int wB)
{

   // 2D Thread ID
   int tx = blockIdx.x * TILE_SIZE + threadIdx.x;
   int ty = blockIdx.y * TILE_SIZE + threadIdx.y;

   // value stores the element that is
   // computed by the thread
   real_t value = 0;
   for (int i = 0; i < wA; ++i)
   {
     real_t elementA = A[ty * wA + i];
     real_t elementB = B[i * wB + tx];
      value += elementA * elementB;
   }

   // Write the matrix to device memory
   // each thread writes one element
   C[ty * wA + tx] = value;
}
#endif

void gpu_mat_mul(real_t* h_A, real_t* h_B, real_t* h_C )
{

    // allocate device memory
    real_t* d_A;
    real_t* d_B;
    real_t* d_C;

    unsigned int size_A = WA * HA;
    unsigned int size_B = WB * HB;
    unsigned int size_C = WC * HC;

    unsigned int mem_size_A = sizeof(real_t) * size_A;
    unsigned int mem_size_B = sizeof(real_t) * size_B;
    unsigned int mem_size_C = sizeof(real_t) * size_C;

    cudaMalloc((void**) &d_A, mem_size_A);
    cudaMalloc((void**) &d_B, mem_size_B);
    cudaMalloc((void**) &d_C, mem_size_C);

    // copy host memory to device*/
    cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
 
    // perform the calculation

    // setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(WC / threads.x, HC / threads.y);
 
    //   execute the kernel
    cudakernel_matrix_mul<<< grid, threads >>>(d_C, d_A, d_B, WA, WB);
 
    // copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
 
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
