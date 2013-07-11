#ifndef kronos_CudaGemm_h
#define kronos_CudaGemm_h

#include "Gemm.h"

namespace kronos {

//------------------------------------------------------------------------------------------

class CudaGemm : public Gemm {

public:

    CudaGemm();

    std::string name() { return "cuda"; }

    void initiate_env();
    void copy_in();
    void compute();
    void copy_out();
    void terminate_env();

private: // members

    real_t* d_A; ///< device memory matrix A
    real_t* d_B; ///< device memory matrix B
    real_t* d_C; ///< device memory matrix C
};

//------------------------------------------------------------------------------------------

} // namespace kronos

#endif
