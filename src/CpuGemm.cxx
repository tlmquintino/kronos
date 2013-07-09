#include "CpuGemm.h"

kronos::CpuGemm::CpuGemm()
{
}

void kronos::CpuGemm::compute()
{
    const size_t M = mm_->m_;
    const size_t K = mm_->k_;
    const size_t N = mm_->n_;

    const real_t* A = &mm_->A.data()[0];
    const real_t* B = &mm_->B.data()[0];
          real_t* C = &mm_->C.data()[0];

    for(size_t i=0;i< M;i++)
    {
      for(size_t j=0;j< N;j++)
      {
        C[i * N + j] = 0.0;
        for(size_t k=0;k< K;k++)
        {
          C[i * N + j] +=  A[i * K + k] * B[k * N +j];
        }
      }
    }
}
