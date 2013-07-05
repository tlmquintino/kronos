#ifndef kronos_ViennaCLGemm_h
#define kronos_ViennaCLGemm_h

#include "Gemm.h"

namespace kronos {

//------------------------------------------------------------------------------------------

class ViennaCLGemm : public Gemm {

public:

    ViennaCLGemm();

    std::string name() { return "viennacl"; }

    void compute();

};

//------------------------------------------------------------------------------------------

} // namespace kronos

#endif
