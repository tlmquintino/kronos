#include "mkl.h"
#include "mkl_cblas.h"

#include "MKLGemm.h"

kronos::MKLGemm::MKLGemm()
{
}

void kronos::MKLGemm::compute()
{
    const real_t alpha  = 1.0;
    const real_t beta   = 0.0;

    const int m = md->A.size1();
    const int k = md->A.size2();
    const int n = md->B.size2();

    const int lda = m; // leading dimension in A, lda>=max(1,m)
    const int ldb = k; // leading dimension in B, ldb>=max(1,k)
    const int ldc = m; // leading dimension in C, ldc>=max(1,m)

    const real_t* A = &md->A.data()[0];
    const real_t* B = &md->B.data()[0];
          real_t* C = &md->C.data()[0];

//    std::cout << "max threads : " << MKL_Get_Max_Threads() << std::endl;
    MKL_Set_Num_Threads(8);
    mkl_domain_set_num_threads ( 8, MKL_BLAS );

    cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
                 m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
