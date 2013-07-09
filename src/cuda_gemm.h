#ifndef cuda_gemm_h
#define cuda_gemm_h

#include "kronos_config.h"

#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

void cuda_gemm( real_t* d_A, real_t* d_B, real_t* d_C, int M, int K, int N );

#endif
