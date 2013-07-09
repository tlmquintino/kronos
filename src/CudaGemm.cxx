#include <cuda.h>
#include <cuda_runtime.h>

#include "CudaUtils.h"

#include "CudaGemm.h"
#include "cuda_gemm.h"

//------------------------------------------------------------------------------------------

kronos::CudaGemm::CudaGemm()
{
}

void kronos::CudaGemm::initiate()
{
    size_A = mm_->A.size1() * mm_->A.size2();
    size_B = mm_->B.size1() * mm_->B.size2();
    size_C = mm_->C.size1() * mm_->C.size2();

    unsigned int mem_size_A = sizeof(real_t) * size_A;
    unsigned int mem_size_B = sizeof(real_t) * size_B;
    unsigned int mem_size_C = sizeof(real_t) * size_C;

    CALL_CUDA( cudaMalloc((void**) &d_A, mem_size_A) );
    CALL_CUDA( cudaMalloc((void**) &d_B, mem_size_B) );
    CALL_CUDA( cudaMalloc((void**) &d_C, mem_size_C) );

    const real_t* A = &mm_->A.data()[0];
    const real_t* B = &mm_->B.data()[0];

    /* copy host memory to device */

    CALL_CUDA( cudaMemcpy(d_A, A, mem_size_A, cudaMemcpyHostToDevice) );
    CALL_CUDA( cudaMemcpy(d_B, B, mem_size_B, cudaMemcpyHostToDevice) );

    copy_into_ += mem_size_A + mem_size_B;
}

void kronos::CudaGemm::compute()
{
    const int M = mm_->m_;
    const int K = mm_->k_;
    const int N = mm_->n_;

    cuda_gemm(d_A,d_B,d_C,M,K,N);
}

void kronos::CudaGemm::terminate()
{
    unsigned int mem_size_C = sizeof(real_t) * size_C;

    real_t* C = &mm_->C.data()[0];

    /* copy result from device to host */

    CALL_CUDA( cudaMemcpy(C, d_C, mem_size_C, cudaMemcpyDeviceToHost) );

    copy_back_ += mem_size_C;

    CALL_CUDA( cudaFree(d_A) );
    CALL_CUDA( cudaFree(d_B) );
    CALL_CUDA( cudaFree(d_C) );
}


