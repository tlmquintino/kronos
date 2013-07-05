#ifndef kronos_CLGemm_h
#define kronos_CLGemm_h

#include "Gemm.h"

namespace kronos {

//------------------------------------------------------------------------------------------

class CLGemm : public Gemm {

public:

    CLGemm();

    std::string name() { return "cl"; }

    void compute();

};

//------------------------------------------------------------------------------------------

} // namespace kronos

#endif
