#ifndef kronos_ViennaCLGemm_h
#define kronos_ViennaCLGemm_h

#include "viennacl/matrix.hpp"

#include "kronos_config.h"

#include "Gemm.h"

namespace kronos {

//------------------------------------------------------------------------------------------

class ViennaCLGemm : public Gemm {

public:

    ViennaCLGemm();

    std::string name() { return "viennacl"; }

    void initiate_env();
    void copy_in();
    void compute();
    void copy_out();
    void terminate_env();

private:

    viennacl::matrix<real_t> d_A;
    viennacl::matrix<real_t> d_B;
    viennacl::matrix<real_t> d_C;

};

//------------------------------------------------------------------------------------------

} // namespace kronos

#endif
