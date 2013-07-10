#include "kronos_config.h"

#include "CublasGemm.h"

//------------------------------------------------------------------------------------------

kronos::CublasGemm::CublasGemm()
{
}

void kronos::CublasGemm::initiate()
{
    int devID = 0;

    CALL_CUDA( cudaSetDevice(devID) );

    CALL_CUDA( cudaGetDevice(&devID) );

    cudaDeviceProp deviceProp;

    CALL_CUDA( cudaGetDeviceProperties(&deviceProp, devID) );

//    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);

    size_A = mm_->A.size1() * mm_->A.size2();
    size_B = mm_->B.size1() * mm_->B.size2();
    size_C = mm_->C.size1() * mm_->C.size2();

    unsigned int mem_size_A = sizeof(real_t) * size_A;
    unsigned int mem_size_B = sizeof(real_t) * size_B;
    unsigned int mem_size_C = sizeof(real_t) * size_C;

    CALL_CUDA( cudaMalloc((void **) &d_A, mem_size_A) );
    CALL_CUDA( cudaMalloc((void **) &d_B, mem_size_B) );
    CALL_CUDA( cudaMalloc((void **) &d_C, mem_size_C) );

    // copy host memory to device

    real_t* A = &mm_->A.data()[0];
    real_t* B = &mm_->B.data()[0];

    CALL_CUDA( cudaMemcpy(d_A, A, mem_size_A, cudaMemcpyHostToDevice) );
    CALL_CUDA( cudaMemcpy(d_B, B, mem_size_B, cudaMemcpyHostToDevice) );

    copy_into_ += mem_size_A + mem_size_B;
}

void kronos::CublasGemm::compute()
{
    const int m = mm_->m_;
    const int k = mm_->k_;
    const int n = mm_->n_;

    const int lda = m; // leading dimension in A, lda>=max(1,m)
    const int ldb = k; // leading dimension in B, ldb>=max(1,k)
    const int ldc = m; // leading dimension in C, ldc>=max(1,m)

    CALL_CUBLAS( cublasCreate(&handle) );

    const real_t alpha = 1.0;
    const real_t beta  = 0.0;

#if USE_DOUBLE

    CALL_CUBLAS( cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc) );

#else

    CALL_CUBLAS( cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc) );

#endif

}

void kronos::CublasGemm::terminate()
{
    unsigned int mem_size_C = sizeof(real_t) * size_C;

    real_t* C = &mm_->C.data()[0];

    /* copy result from device to host */

    CALL_CUDA( cudaMemcpy(C, d_C, mem_size_C, cudaMemcpyDeviceToHost) );

    copy_back_ += mem_size_C;

    CALL_CUBLAS( cublasDestroy(handle) );

    CALL_CUDA( cudaFree(d_A) );
    CALL_CUDA( cudaFree(d_B) );
    CALL_CUDA( cudaFree(d_C) );

    CALL_CUDA( cudaDeviceReset() );
}
