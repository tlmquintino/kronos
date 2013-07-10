#ifndef kronos_CudaGemm_h
#define kronos_CudaGemm_h

#include "Gemm.h"

namespace kronos {

//------------------------------------------------------------------------------------------

class CudaGemm : public Gemm {

public:

    CudaGemm();

    std::string name() { return "cuda"; }

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
};

//------------------------------------------------------------------------------------------

} // namespace kronos

#endif
