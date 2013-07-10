#include "viennacl/linalg/prod.hpp"

#include "ViennaCLGemm.h"

kronos::ViennaCLGemm::ViennaCLGemm()
{
}

void kronos::ViennaCLGemm::initiate()
{
    const size_t M = mm_->m_;
    const size_t K = mm_->k_;
    const size_t N = mm_->n_;

    MData::matrix_t& A = mm_->A;
    MData::matrix_t& B = mm_->B;

    d_A.resize( M, K );
    d_B.resize( K, N );
    d_C.resize( M, N );

    viennacl::copy( A,  d_A );
    viennacl::copy( B,  d_B );

    copy_into_ += A.size1() * A.size2() * sizeof(real_t) +
                  B.size1() * B.size2() * sizeof(real_t);
}

void kronos::ViennaCLGemm::compute()
{
    d_C = viennacl::linalg::prod( d_A, d_B );
}

void kronos::ViennaCLGemm::terminate()
{
    MData::matrix_t& C = mm_->C;

    viennacl::copy( d_C,  C );

    copy_back_ += C.size1() * C.size2() * sizeof(real_t);
}
