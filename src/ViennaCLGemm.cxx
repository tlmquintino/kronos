#include "viennacl/linalg/prod.hpp"

#include "ViennaCLGemm.h"

kronos::ViennaCLGemm::ViennaCLGemm()
{
}

void kronos::ViennaCLGemm::initiate_env()
{
    const size_t M = md->m_;
    const size_t K = md->k_;
    const size_t N = md->n_;

    MData::matrix_t& A = md->A;
    MData::matrix_t& B = md->B;

    d_A.resize( M, K );
    d_B.resize( K, N );
    d_C.resize( M, N );

    viennacl::copy( A,  d_A );
}

void kronos::ViennaCLGemm::copy_in()
{
    MData::matrix_t& B = md->B;

    viennacl::copy( B, d_B );

    copy_into_ += size_B * sizeof(real_t);
}

void kronos::ViennaCLGemm::compute()
{
    d_C = viennacl::linalg::prod( d_A, d_B );
}

void kronos::ViennaCLGemm::copy_out()
{
    MData::matrix_t& C = md->C;

    viennacl::copy( d_C,  C );

    copy_back_ += size_C * sizeof(real_t);
}

void kronos::ViennaCLGemm::terminate_env()
{
    d_A.clear();
    d_B.clear();
    d_C.clear();
}
