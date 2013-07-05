#ifndef matrix_mult_h
#define matrix_mult_h

#include "kronos_config.h"

#ifdef __cplusplus
extern "C" {
#endif

void gpu_mat_mul(real_t* h_A, real_t* h_B, real_t* h_C );

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // matrix_mult_h
