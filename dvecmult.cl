#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#else
#error no double extension available
#endif

__kernel void vecmult(__global double* a, __global double* b, __global double* c)
{
    unsigned int i = get_global_id(0);

    c[i] = a[i] * b[i];
}
