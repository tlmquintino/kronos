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

    void initiate_env();
    void copy_in();
    void compute();
    void copy_out();
    void terminate_env();

private: // members

    real_t* d_A; ///< device memory matrix A
    real_t* d_B; ///< device memory matrix B
    real_t* d_C; ///< device memory matrix C

    real_t* p_B; ///< pinned host memory matrix B
    real_t* p_C; ///< pinned host memory matrix C

    cublasHandle_t handle;
    cudaError_t error;
};

//------------------------------------------------------------------------------------------

} // namespace kronos

#endif
