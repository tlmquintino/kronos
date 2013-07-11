#include <cuda.h>
#include <cuda_runtime.h>

#include "CudaUtils.h"

#include "CudaGemm.h"
#include "cuda_gemm.h"

//------------------------------------------------------------------------------------------

kronos::CudaGemm::CudaGemm()
{
}

void kronos::CudaGemm::initiate_env()
{
    unsigned int mem_size_A = sizeof(real_t) * size_A;
    unsigned int mem_size_B = sizeof(real_t) * size_B;
    unsigned int mem_size_C = sizeof(real_t) * size_C;

    CALL_CUDA( cudaMalloc((void**) &d_A, mem_size_A) );
    CALL_CUDA( cudaMalloc((void**) &d_B, mem_size_B) );
    CALL_CUDA( cudaMalloc((void**) &d_C, mem_size_C) );

    const real_t* A = &md->A.data()[0];
    const real_t* B = &md->B.data()[0];

    CALL_CUDA( cudaMemcpy(d_A, A, mem_size_A, cudaMemcpyHostToDevice) );
}

void kronos::CudaGemm::copy_in()
{
    size_B = md->B.size1() * md->B.size2();

    unsigned int mem_size_B = sizeof(real_t) * size_B;

    const real_t* B = &md->B.data()[0];

    CALL_CUDA( cudaMemcpy(d_B, B, mem_size_B, cudaMemcpyHostToDevice) );

    copy_into_ += mem_size_B;
}

void kronos::CudaGemm::compute()
{
    const int M = md->m_;
    const int K = md->k_;
    const int N = md->n_;

    cuda_gemm(d_A,d_B,d_C,M,K,N);
}

void kronos::CudaGemm::copy_out()
{
    unsigned int mem_size_C = sizeof(real_t) * size_C;

    real_t* C = &md->C.data()[0];

    CALL_CUDA( cudaMemcpy(C, d_C, mem_size_C, cudaMemcpyDeviceToHost) );

    copy_back_ += mem_size_C;
}

void kronos::CudaGemm::terminate_env()
{
    CALL_CUDA( cudaFree(d_A) );
    CALL_CUDA( cudaFree(d_B) );
    CALL_CUDA( cudaFree(d_C) );
}


