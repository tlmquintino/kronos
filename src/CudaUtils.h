#ifndef kronos_CublasUtils_h
#define kronos_CublasUtils_h

#include <cstdio>

#include <cuda_runtime.h>
#include <cublas_v2.h>

//------------------------------------------------------------------------------------------

#define CALL_CUDA(e) \
{ cudaError_t error; \
  if( ( error = e ) != cudaSuccess ) printf("%s failed with error code %d @ %s +%d\n", #e, error, __FILE__, __LINE__), exit(EXIT_FAILURE); \
}

#define CALL_CUBLAS(e) \
{ cublasStatus_t error; \
  if( ( error = e ) != CUBLAS_STATUS_SUCCESS ) printf("%s failed with error code %d @ %s +%d\n", #e, error, __FILE__, __LINE__), exit(EXIT_FAILURE); \
}

//------------------------------------------------------------------------------------------

#endif
