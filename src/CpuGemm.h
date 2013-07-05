#ifndef kronos_CpuGemm_h
#define kronos_CpuGemm_h

#include "Gemm.h"

namespace kronos {

//------------------------------------------------------------------------------------------

class CpuGemm : public Gemm {

public:

    CpuGemm();

    std::string name() { return "cpu"; }

    void compute();

};

//------------------------------------------------------------------------------------------

} // namespace kronos

#endif
