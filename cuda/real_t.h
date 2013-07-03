#ifndef cuda_real_t_h
#define cuda_real_t_h

#if USE_DOUBLE
typedef double real_t;
#else
typedef float  real_t;
#endif

#endif
