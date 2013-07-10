#ifndef kronos_CLGemm_h
#define kronos_CLGemm_h

#include "kronos_opencl.h"

#include "Gemm.h"

namespace kronos {

//------------------------------------------------------------------------------------------

class CLGemm : public Gemm {

public:

    CLGemm();

    std::string name() { return "cl"; }

    void copy_into();
    void compute();
    void copy_out();

};

//------------------------------------------------------------------------------------------

} // namespace kronos

#endif
