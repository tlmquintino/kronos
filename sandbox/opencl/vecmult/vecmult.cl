#if USE_DOUBLE

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#error no double extension available
#endif
#endif

// double
typedef double   real_t;
typedef double2  real2_t;
#define PI       3.14159265358979323846

#else

// float
typedef float    real_t;
typedef float2   real2_t;
#define PI       3.14159265359f

#endif

__kernel void vecmult(__global real_t* a, __global real_t* b, __global real_t* c)
{
    unsigned int i = get_global_id(0);

    c[i] = a[i] * b[i];
}
