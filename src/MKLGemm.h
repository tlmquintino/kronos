#ifndef kronos_MKLGemm_h
#define kronos_MKLGemm_h

#include "Gemm.h"

namespace kronos {

//------------------------------------------------------------------------------------------

class MKLGemm : public Gemm {

public:

    MKLGemm();

    std::string name() { return "mkl"; }

    void compute();

};

//------------------------------------------------------------------------------------------

} // namespace kronos

#endif
