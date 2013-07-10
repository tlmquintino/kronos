#include "BlasGemm.h"

extern "C" {

extern void sgemm_( char *transa, char *transb,
                    int *m, int *n, int *k,
                    float *alpha, float *a, int *lda,
                    float *b, int *ldb,
                    float *beta,
                    float *c, int *ldc );

extern void dgemm_( char *transa, char *transb,
                    int *m, int *n, int *k,
                    double *alpha, double *a, int *lda,
                    double *b, int *ldb,
                    double *beta,
                    double *c, int *ldc );
}

kronos::BlasGemm::BlasGemm()
{
}

void kronos::BlasGemm::compute()
{
    real_t alpha  = 1.0;
    real_t beta   = 0.0;

    int m = mm_->m_;
    int k = mm_->k_;
    int n = mm_->n_;

    int lda = m; // leading dimension in A, lda>=max(1,m)
    int ldb = k; // leading dimension in B, ldb>=max(1,k)
    int ldc = m; // leading dimension in C, ldc>=max(1,m)

    real_t* A = &mm_->A.data()[0];
    real_t* B = &mm_->B.data()[0];
    real_t* C = &mm_->C.data()[0];

//    for( int i = 0; i < m*n; ++i )
//        C[i] = 0.;

#if USE_DOUBLE
    dgemm_( "n", "n", &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
#else
    sgemm_( "n", "n", &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
#endif

}
