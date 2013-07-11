#include "CpuGemm.h"

kronos::CpuGemm::CpuGemm()
{
}

void kronos::CpuGemm::compute()
{
    MData::matrix_t& A = md->A;
    MData::matrix_t& B = md->B;
    MData::matrix_t& C = md->C;

    /* rely on boost::ublas for gemm */

    C = prod( A , B );
}
