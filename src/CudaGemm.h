#ifndef kronos_CudaGemm_h
#define kronos_CudaGemm_h

#include "Gemm.h"

namespace kronos {

//------------------------------------------------------------------------------------------

class CudaGemm : public Gemm {

public:

    CudaGemm();

    std::string name() { return "cuda"; }

    void compute();

};

//------------------------------------------------------------------------------------------

} // namespace kronos

#endif
