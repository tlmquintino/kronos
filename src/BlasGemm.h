#ifndef kronos_BlasGemm_h
#define kronos_BlasGemm_h

#include "Gemm.h"

namespace kronos {

//------------------------------------------------------------------------------------------

class BlasGemm : public Gemm {

public:

    BlasGemm();

    std::string name() { return "blas"; }

    void compute();

};

//------------------------------------------------------------------------------------------

} // namespace kronos

#endif
