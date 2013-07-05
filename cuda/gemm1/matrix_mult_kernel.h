#ifndef matrix_mult_kernel_h
#define matrix_mult_kernel_h
 
#include <stdio.h>

#include "kronos_config.h"
#include "matrix_sizes.h"

// CUDA Kernel
__global__ void
matrixMul( real_t* C, real_t* A, real_t* B, int wA, int wB)
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


