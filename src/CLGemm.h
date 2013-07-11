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

    void initiate_env();
    void copy_in();
    void compute();
    void copy_out();
    void terminate_env();

};

//------------------------------------------------------------------------------------------

} // namespace kronos

#endif
