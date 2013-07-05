#ifndef kronos_CublasGemm_h
#define kronos_CublasGemm_h

#include "Gemm.h"

namespace kronos {

//------------------------------------------------------------------------------------------

class CublasGemm : public Gemm {

public:

    CublasGemm();

    std::string name() { return "cublas"; }

    void compute();

};

//------------------------------------------------------------------------------------------

} // namespace kronos

#endif
