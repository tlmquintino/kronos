#include "BlasGemm.h"

extern "C" {
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

    real_t* A = &mm_->A.data()[0];
    real_t* B = &mm_->B.data()[0];
    real_t* C = &mm_->C.data()[0];

    dgemm_( "n", "n", &m, &n, &k, &alpha, A, &k, B, &n, &beta, C, &n );

}
