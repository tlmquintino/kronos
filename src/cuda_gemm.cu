#include "cuda_gemm.h"

// CUDA GEMM kernel
// multiply two matrices C = A * B
__global__ void
cuda_gemm_kernel( real_t* C, real_t* A, real_t* B, int wA, int wB)
{
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Index of the first sub-matrix of A processed
  // by the block
  int aBegin = wA * BLOCK_SIZE * by;

  // Index of the last sub-matrix of A processed
  // by the block
  int aEnd   = aBegin + wA - 1;

  // Step size used to iterate through the
  // sub-matrices of A
  int aStep  = BLOCK_SIZE;

  // Index of the first sub-matrix of B processed
  // by the block
  int bBegin = BLOCK_SIZE * bx;

  // Step size used to iterate through the
  // sub-matrices of B
  int bStep  = BLOCK_SIZE * wB;

  real_t Csub = 0.;

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int a = aBegin, b = bBegin;
           a <= aEnd;
           a += aStep, b += bStep)
  {

    // Declaration of the shared memory array As
    // used to store the sub-matrix of A
    __shared__ real_t As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs
    // used to store the sub-matrix of B
    __shared__ real_t Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Load the matrices from global memory
    // to shared memory; each thread loads
    // one element of each matrix
    As[ty][tx] = A[a + wA * ty + tx];
    Bs[ty][tx] = B[b + wB * ty + tx];

    // Synchronize to make sure the matrices
    // are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
    Csub = 0.;
    for (int k = 0; k < BLOCK_SIZE; ++k)
      Csub += As[ty][k] * Bs[k][tx];

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  C[c + wB * ty + tx] = Csub;
}


void cuda_gemm( real_t* d_A, real_t* d_B, real_t* d_C, int M, int K, int N )
{

    // setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid( M / threads.x, N / threads.y );

    //   execute the kernel
    cuda_gemm_kernel<<< grid, threads >>>(d_C, d_A, d_B, K, N);

}
