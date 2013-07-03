#include <cstdio>
#include <cstdlib>
#include <cassert>

#include <iostream>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "mmcublas.h"

void mmcublas_init()
{
    cudaError_t error;
    int devID = 0;

    // default device 0
    error = cudaSetDevice(devID);

    if (error != cudaSuccess)
    {
        printf("cudaSetDevice returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // get number of SMs on this GPU
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaDeviceProp deviceProp;

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
}

void mmcublas_dgemm(real_t* A, real_t* B, real_t* C )
{
    cudaDeviceProp deviceProp;
    cudaError_t error;

    error = cudaGetDeviceProperties(&deviceProp, 0);

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // use a larger block size for Fermi and above
    int block_size = (deviceProp.major < 2) ? 16 : 32;

    unsigned int size_A = WA * HA;
    unsigned int size_B = WB * HB;
    unsigned int size_C = WC * HC;

    unsigned int mem_size_A = sizeof(real_t) * size_A;
    unsigned int mem_size_B = sizeof(real_t) * size_B;
    unsigned int mem_size_C = sizeof(real_t) * size_C;

    real_t *d_A, *d_B, *d_C;

    error = cudaMalloc((void **) &d_A, mem_size_A);
    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &d_B, mem_size_B);
    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_B returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // copy host memory to device
    error = cudaMemcpy(d_A, A, mem_size_A, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy d_A A returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(d_B, B, mem_size_B, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy d_B B returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &d_C, mem_size_C);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_C returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // CUBLAS version 2.0
    {
        cublasHandle_t handle;

        cublasStatus_t ret;

        ret = cublasCreate(&handle);

        if (ret != CUBLAS_STATUS_SUCCESS)
        {
            printf("cublasCreate returned error code %d, line(%d)\n", ret, __LINE__);
            exit(EXIT_FAILURE);
        }

        const real_t alpha = 1.0f;
        const real_t beta  = 1.0f;

        ret = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, WB, HA, WA, &alpha, d_B, WB, d_A, WA, &beta, d_C, WA);

        if (ret != CUBLAS_STATUS_SUCCESS)
        {
            printf("cublasDgemm returned error code %d, line(%d)\n", ret, __LINE__);
            exit(EXIT_FAILURE);
        }


        // copy result from device to host
        error = cudaMemcpy(C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

        if (error != cudaSuccess)
        {
            printf("cudaMemcpy C returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

        if (cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS)
        {
            printf("%s", "cublasDestroy() error!\n");
            exit(EXIT_FAILURE);
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaDeviceReset();
}


