#ifndef mmcublas_h
#define mmcublas_h

#include "kronos_config.h"
#include "matrix_sizes.h"

void mmcublas_init();
void mmcublas_dgemm(real_t* A, real_t* B, real_t* C );

#endif
