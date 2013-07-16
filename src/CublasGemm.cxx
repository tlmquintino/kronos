#include "kronos_config.h"

#include "CublasGemm.h"

//------------------------------------------------------------------------------------------

#define PINNED_MEM

//#ifdef PINNED_MEM
//  #define OVERLAP_COMP
//#endif

//------------------------------------------------------------------------------------------

kronos::CublasGemm::CublasGemm()
{
}

void kronos::CublasGemm::initiate_env()
{
    int devID = 0;

    CALL_CUDA( cudaSetDevice(devID) );

    CALL_CUDA( cudaGetDevice(&devID) );

    cudaDeviceProp deviceProp;

    CALL_CUDA( cudaGetDeviceProperties(&deviceProp, devID) );

//        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);

    CALL_CUDA( cudaMallocHost((void**)&p_B, mem_size_B) );
    CALL_CUDA( cudaMallocHost((void**)&p_C, mem_size_C) );

    CALL_CUDA( cudaMalloc((void**) &d_A, mem_size_A) );
    CALL_CUDA( cudaMalloc((void**) &d_B, mem_size_B) );
    CALL_CUDA( cudaMalloc((void**) &d_C, mem_size_C) );

    CALL_CUBLAS( cublasCreate(&handle) );

}

void kronos::CublasGemm::pre_process()
{
    real_t* A = &md->A.data()[0];
    CALL_CUDA( cudaMemcpy(d_A, A, mem_size_A, cudaMemcpyHostToDevice) );

#ifdef PINNED_MEM
    real_t* B = &md->B.data()[0];
    ::memcpy(p_B,B,mem_size_B);
#endif
}

void kronos::CublasGemm::copy_in()
{
#ifdef PINNED_MEM
    CALL_CUDA( cudaMemcpy(d_B, p_B, mem_size_B, cudaMemcpyHostToDevice) );
#else
    real_t* B = &md->B.data()[0];
    CALL_CUDA( cudaMemcpy(d_B, B, mem_size_B, cudaMemcpyHostToDevice) );
#endif

    copy_into_ += mem_size_B;
}

void kronos::CublasGemm::compute()
{
    const real_t alpha = 1.0;
    const real_t beta  = 0.0;

    const int m = md->A.size1();
    const int k = md->A.size2();
    const int n = md->B.size2();

    const int lda = m; // leading dimension in A, lda>=max(1,m)
    const int ldb = k; // leading dimension in B, ldb>=max(1,k)
    const int ldc = m; // leading dimension in C, ldc>=max(1,m)

#if USE_DOUBLE

    CALL_CUBLAS( cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc) );

#else

    CALL_CUBLAS( cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc) );

#endif

    CALL_CUDA( cudaThreadSynchronize() ); /* ensure we finish the computation before taking timings */
}

void kronos::CublasGemm::copy_out()
{
#ifndef PINNED_MEM
    real_t* C = &md->C.data()[0];
    CALL_CUDA( cudaMemcpy(C, d_C, mem_size_C, cudaMemcpyDeviceToHost) );
#else
    CALL_CUDA( cudaMemcpy(p_C, d_C, mem_size_C, cudaMemcpyDeviceToHost) );
#endif
    copy_back_ += mem_size_C;
}

void kronos::CublasGemm::post_process()
{
#ifdef PINNED_MEM
    real_t* C = &md->C.data()[0];
    ::memcpy(C,p_C,mem_size_C);
#endif
}

void kronos::CublasGemm::terminate_env()
{
    CALL_CUBLAS( cublasDestroy(handle) );

    CALL_CUDA( cudaFree(d_A) );
    CALL_CUDA( cudaFree(d_B) );
    CALL_CUDA( cudaFree(d_C) );

    cudaFreeHost(p_B);
    cudaFreeHost(p_C);

    CALL_CUDA( cudaDeviceReset() );
}
