#pragma once
#ifdef __DEVICE__
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define LDAS 65
#define LDBS 17

__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void dgemm(int n, int m, int k, double alpha, __global double* a, int lda, __global double* b, int ldb, double beta, __global double* c, int ldc)
{
  double c0_0 = 0.0;
  double c0_16 = 0.0;
  double c0_32 = 0.0;
  double c0_48 = 0.0;
  double c16_0 = 0.0;
  double c16_16 = 0.0;
  double c16_32 = 0.0;
  double c16_48 = 0.0;
  double c32_0 = 0.0;
  double c32_16 = 0.0;
  double c32_32 = 0.0;
  double c32_48 = 0.0;
  double c48_0 = 0.0;
  double c48_16 = 0.0;
  double c48_32 = 0.0;
  double c48_48 = 0.0;

  double a0;
  double a16;
  double a32;
  double a48;

  double b0;
  double b16;
  double b32;
  double b48;

  __local double aShared[LDAS * 16];
  __local double bShared[LDBS * 64];

  const int rowBlock = get_local_id(0);
  const int colBlock = get_local_id(1);

  const int rowC = get_group_id(0) * 64;
  const int colC = get_group_id(1) * 64;

  __global double* blockC;

  for(int i = 0; i < k; i+= 16)
  {
    #pragma unroll 4
    for(int l = 0; l < 64; l+= 16)
    {
      aShared[rowBlock + l + LDAS * colBlock] = a[rowC + rowBlock + l + (colBlock + i) * lda];
      bShared[colBlock + LDBS * (rowBlock + l)] = b[colBlock + i + (colC + rowBlock + l) * ldb];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for(int l = 0; l <16; l++)
    {
      a0 = aShared[rowBlock + 0 + LDAS * l];
      a16 = aShared[rowBlock + 16 + LDAS * l];
      a32 = aShared[rowBlock + 32 + LDAS * l];
      a48 = aShared[rowBlock + 48 + LDAS * l];

      b0 = bShared[l + LDBS * (colBlock + 0)];
      b16 = bShared[l + LDBS * (colBlock + 16)];
      b32 = bShared[l + LDBS * (colBlock + 32)];
      b48 = bShared[l + LDBS * (colBlock + 48)];

      c0_0 = fma(a0, b0, c0_0);
      c0_16 = fma(a0, b16, c0_16);
      c0_32 = fma(a0, b32, c0_32);
      c0_48 = fma(a0, b48, c0_48);
      c16_0 = fma(a16, b0, c16_0);
      c16_16 = fma(a16, b16, c16_16);
      c16_32 = fma(a16, b32, c16_32);
      c16_48 = fma(a16, b48, c16_48);
      c32_0 = fma(a32, b0, c32_0);
      c32_16 = fma(a32, b16, c32_16);
      c32_32 = fma(a32, b32, c32_32);
      c32_48 = fma(a32, b48, c32_48);
      c48_0 = fma(a48, b0, c48_0);
      c48_16 = fma(a48, b16, c48_16);
      c48_32 = fma(a48, b32, c48_32);
      c48_48 = fma(a48, b48, c48_48);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  blockC = &c[rowC + ldc * (colC + colBlock)];

  if(rowC + 63 < m && colC + 63 < n)
  {
    blockC[rowBlock + 0] = alpha * c0_0 + beta * blockC[rowBlock + 0];
    blockC[rowBlock + 16] = alpha * c16_0 + beta * blockC[rowBlock + 16];
    blockC[rowBlock + 32] = alpha * c32_0 + beta * blockC[rowBlock + 32];
    blockC[rowBlock + 48] = alpha * c48_0 + beta * blockC[rowBlock + 48];

    blockC += 16 * ldc;

    blockC[rowBlock + 0] = alpha * c0_16 + beta * blockC[rowBlock + 0];
    blockC[rowBlock + 16] = alpha * c16_16 + beta * blockC[rowBlock + 16];
    blockC[rowBlock + 32] = alpha * c32_16 + beta * blockC[rowBlock + 32];
    blockC[rowBlock + 48] = alpha * c48_16 + beta * blockC[rowBlock + 48];

    blockC += 16 * ldc;

    blockC[rowBlock + 0] = alpha * c0_32 + beta * blockC[rowBlock + 0];
    blockC[rowBlock + 16] = alpha * c16_32 + beta * blockC[rowBlock + 16];
    blockC[rowBlock + 32] = alpha * c32_32 + beta * blockC[rowBlock + 32];
    blockC[rowBlock + 48] = alpha * c48_32 + beta * blockC[rowBlock + 48];

    blockC += 16 * ldc;

    blockC[rowBlock + 0] = alpha * c0_48 + beta * blockC[rowBlock + 0];
    blockC[rowBlock + 16] = alpha * c16_48 + beta * blockC[rowBlock + 16];
    blockC[rowBlock + 32] = alpha * c32_48 + beta * blockC[rowBlock + 32];
    blockC[rowBlock + 48] = alpha * c48_48 + beta * blockC[rowBlock + 48];

    blockC += 16 * ldc;

  }

}

#else
#define DGEMMTHREADROWS 16
#define DGEMMTHREADCOLS 16
#define DGEMMROWREP 4
#define DGEMMCOLREP 4
#endif
