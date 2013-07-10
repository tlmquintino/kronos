#ifndef kronos_CublasGemm_h
#define kronos_CublasGemm_h

#include "CudaUtils.h"
#include "Gemm.h"

namespace kronos {

//------------------------------------------------------------------------------------------

class CublasGemm : public Gemm {

public:

    CublasGemm();

    std::string name() { return "cublas"; }

    void copy_into();
    void compute();
    void copy_out();

private: // members

    unsigned int size_A;
    unsigned int size_B;
    unsigned int size_C;

    real_t* d_A; ///< device memory matrix A
    real_t* d_B; ///< device memory matrix B
    real_t* d_C; ///< device memory matrix C

    cublasHandle_t handle;
    cudaError_t error;
};

//------------------------------------------------------------------------------------------

} // namespace kronos

#endif
