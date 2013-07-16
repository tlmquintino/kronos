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

    const real_t* B = &md->B.data()[0];

    CALL_CUDA( cudaMemcpy(d_B, B, mem_size_B, cudaMemcpyHostToDevice) );

    copy_into_ += mem_size_B;
}

void kronos::CudaGemm::compute()
{
    const int m = md->A.size1();
    const int k = md->A.size2();
    const int n = md->B.size2();

    cuda_gemm(d_A,d_B,d_C,m,k,n);
}

void kronos::CudaGemm::copy_out()
{
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


