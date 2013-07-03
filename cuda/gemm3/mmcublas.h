#ifndef mmcublas_h
#define mmcublas_h

#include "real_t.h"
#include "matrix_sizes.h"

void mmcublas_init();
void mmcublas_dgemm(real_t* A, real_t* B, real_t* C );

#endif
