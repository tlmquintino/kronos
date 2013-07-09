#include "mkl.h"
#include "mkl_cblas.h"

#include "MKLGemm.h"

kronos::MKLGemm::MKLGemm()
{
}

void kronos::MKLGemm::compute()
{
    const real_t alpha = 1.0;
    const real_t beta = 1.0;

    const int m = mm_->m_;
    const int k = mm_->k_;
    const int n = mm_->n_;

    const real_t* A = &mm_->A.data()[0];
    const real_t* B = &mm_->B.data()[0];
          real_t* C = &mm_->C.data()[0];

//    std::cout << "max threads : " << MKL_Get_Max_Threads() << std::endl;
//    MKL_Set_Num_Threads(8);
//    mkl_domain_set_num_threads ( 4, MKL_BLAS );

    cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 m, n, k, alpha, A, k, B, n, beta, C, n);
}
