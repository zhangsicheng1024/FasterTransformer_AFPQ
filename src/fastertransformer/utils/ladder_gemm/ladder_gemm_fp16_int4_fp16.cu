#include <cuda_runtime.h>
#include <assert.h>
#include "ladder_kernel.h"
#include "mma.h"

// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y)
{
    unsigned v0 = *((unsigned short *)&x);
    unsigned v1 = *((unsigned short *)&y);
    return (v1 << 16) | v0;
}

 __global__ void __launch_bounds__(128) cutlass_kernel_fp16_int4_fp16_m1n12288k4096_nt_4x8(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ C) {
            int8_t* B = ((int8_t *)QB);
	 half* Scales = (half *)((int8_t *)QB + 25165824);
	 int8_t* Zeros = ((int8_t *)QB + 25165824 + 786432);                 
            // const dim3 GridDim(3072, 1, 1);
            // const dim3 BlockDim(32, 4, 1);
            // cutlass_kernel_fp16_int4_fp16_m1n12288k4096_nt_4x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
  half in_thread_C_local[1];
  half A_local[8];
  signed char Zeros_local[1];
  half red_buf0[1];
  in_thread_C_local[0] = __float2half_rn(0.000000e+00f);
  for (int k_0 = 0; k_0 < 16; ++k_0) {
    *(uint4*)(A_local + 0) = *(uint4*)(A + ((k_0 * 256) + (((int)threadIdx.x) * 8)));
    Zeros_local[0] = Zeros[(((k_0 * 12288) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.y))];
    for (int k_2 = 0; k_2 < 8; ++k_2) {
      in_thread_C_local[0] = (in_thread_C_local[0] + (A_local[k_2] * (((half)(((B[(((((((int)blockIdx.x) * 8192) + (((int)threadIdx.y) * 2048)) + (k_0 * 128)) + (((int)threadIdx.x) * 4)) + (k_2 >> 1))] >> ((signed char)((k_2 & 1) * 4))) & (signed char)15) - ((Zeros_local[0] >> ((signed char)((k_2 & 1) * 4))) & (signed char)15))) * Scales[((((k_0 * 24576) + ((((int)threadIdx.x) >> 4) * 12288)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.y))])));
    }
  }
  uint mask[1];
  half t0[1];
  red_buf0[0] = in_thread_C_local[0];
  mask[0] = (__activemask() & ((uint)(0 << (((int)threadIdx.y) * 32))));
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 16, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 8, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf0[0] = __shfl_sync(mask[0], red_buf0[0], (((int)threadIdx.y) * 32), 32);
  C[((((int)blockIdx.x) * 4) + ((int)threadIdx.y))] = red_buf0[0];
}



 __global__ void __launch_bounds__(128) cutlass_kernel_fp16_int4_fp16_m1n4096k4096_nt_4x8(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ C) {
            int8_t* B = ((int8_t *)QB);
	 half* Scales = (half *)((int8_t *)QB + 8388608);
	 int8_t* Zeros = ((int8_t *)QB + 8388608 + 262144);                 
            // const dim3 GridDim(1024, 1, 1);
            // const dim3 BlockDim(32, 4, 1);
            // cutlass_kernel_fp16_int4_fp16_m1n4096k4096_nt_4x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
  half in_thread_C_local[1];
  half A_local[8];
  signed char Zeros_local[1];
  half red_buf0[1];
  in_thread_C_local[0] = __float2half_rn(0.000000e+00f);
  for (int k_0 = 0; k_0 < 16; ++k_0) {
    *(uint4*)(A_local + 0) = *(uint4*)(A + ((k_0 * 256) + (((int)threadIdx.x) * 8)));
    Zeros_local[0] = Zeros[(((k_0 * 4096) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.y))];
    for (int k_2 = 0; k_2 < 8; ++k_2) {
      in_thread_C_local[0] = (in_thread_C_local[0] + (A_local[k_2] * (((half)(((B[(((((((int)blockIdx.x) * 8192) + (((int)threadIdx.y) * 2048)) + (k_0 * 128)) + (((int)threadIdx.x) * 4)) + (k_2 >> 1))] >> ((signed char)((k_2 & 1) * 4))) & (signed char)15) - ((Zeros_local[0] >> ((signed char)((k_2 & 1) * 4))) & (signed char)15))) * Scales[((((k_0 * 8192) + ((((int)threadIdx.x) >> 4) * 4096)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.y))])));
    }
  }
  uint mask[1];
  half t0[1];
  red_buf0[0] = in_thread_C_local[0];
  mask[0] = (__activemask() & ((uint)(0 << (((int)threadIdx.y) * 32))));
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 16, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 8, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf0[0] = __shfl_sync(mask[0], red_buf0[0], (((int)threadIdx.y) * 32), 32);
  C[((((int)blockIdx.x) * 4) + ((int)threadIdx.y))] = red_buf0[0];
}



 __global__ void __launch_bounds__(128) cutlass_kernel_fp16_int4_fp16_m1n4096k11008_nt_4x8(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ C) {
            int8_t* B = ((int8_t *)QB);
	 half* Scales = (half *)((int8_t *)QB + 22544384);
	 int8_t* Zeros = ((int8_t *)QB + 22544384 + 704512);                 
            // const dim3 GridDim(1024, 1, 1);
            // const dim3 BlockDim(32, 4, 1);
            // cutlass_kernel_fp16_int4_fp16_m1n4096k11008_nt_4x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
  half in_thread_C_local[1];
  half A_local[8];
  signed char Zeros_local[1];
  half red_buf0[1];
  in_thread_C_local[0] = __float2half_rn(0.000000e+00f);
  for (int k_0 = 0; k_0 < 43; ++k_0) {
    *(uint4*)(A_local + 0) = *(uint4*)(A + ((k_0 * 256) + (((int)threadIdx.x) * 8)));
    if (k_0 < 40) {
      Zeros_local[0] = Zeros[(((k_0 * 4096) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.y))];
    }
    for (int k_2 = 0; k_2 < 8; ++k_2) {
      in_thread_C_local[0] = (in_thread_C_local[0] + (A_local[k_2] * (((half)(((B[(((((((int)blockIdx.x) * 22016) + (((int)threadIdx.y) * 5504)) + (k_0 * 128)) + (((int)threadIdx.x) * 4)) + (k_2 >> 1))] >> ((signed char)((k_2 & 1) * 4))) & (signed char)15) - ((Zeros_local[0] >> ((signed char)((k_2 & 1) * 4))) & (signed char)15))) * Scales[((((k_0 * 8192) + ((((int)threadIdx.x) >> 4) * 4096)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.y))])));
    }
  }
  uint mask[1];
  half t0[1];
  red_buf0[0] = in_thread_C_local[0];
  mask[0] = (__activemask() & ((uint)(0 << (((int)threadIdx.y) * 32))));
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 16, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 8, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf0[0] = __shfl_sync(mask[0], red_buf0[0], (((int)threadIdx.y) * 32), 32);
  C[((((int)blockIdx.x) * 4) + ((int)threadIdx.y))] = red_buf0[0];
}



 __global__ void __launch_bounds__(128) cutlass_kernel_fp16_int4_fp16_m1n11008k4096_nt_4x8(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ C) {
            int8_t* B = ((int8_t *)QB);
	 half* Scales = (half *)((int8_t *)QB + 22544384);
	 int8_t* Zeros = ((int8_t *)QB + 22544384 + 704512);                 
            // const dim3 GridDim(2752, 1, 1);
            // const dim3 BlockDim(32, 4, 1);
            // cutlass_kernel_fp16_int4_fp16_m1n11008k4096_nt_4x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
  half in_thread_C_local[1];
  half A_local[8];
  signed char Zeros_local[1];
  half red_buf0[1];
  in_thread_C_local[0] = __float2half_rn(0.000000e+00f);
  for (int k_0 = 0; k_0 < 16; ++k_0) {
    *(uint4*)(A_local + 0) = *(uint4*)(A + ((k_0 * 256) + (((int)threadIdx.x) * 8)));
    Zeros_local[0] = Zeros[(((k_0 * 11008) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.y))];
    for (int k_2 = 0; k_2 < 8; ++k_2) {
      in_thread_C_local[0] = (in_thread_C_local[0] + (A_local[k_2] * (((half)(((B[(((((((int)blockIdx.x) * 8192) + (((int)threadIdx.y) * 2048)) + (k_0 * 128)) + (((int)threadIdx.x) * 4)) + (k_2 >> 1))] >> ((signed char)((k_2 & 1) * 4))) & (signed char)15) - ((Zeros_local[0] >> ((signed char)((k_2 & 1) * 4))) & (signed char)15))) * Scales[((((k_0 * 22016) + ((((int)threadIdx.x) >> 4) * 11008)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.y))])));
    }
  }
  uint mask[1];
  half t0[1];
  red_buf0[0] = in_thread_C_local[0];
  mask[0] = (__activemask() & ((uint)(0 << (((int)threadIdx.y) * 32))));
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 16, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 8, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf0[0] = __shfl_sync(mask[0], red_buf0[0], (((int)threadIdx.y) * 32), 32);
  C[((((int)blockIdx.x) * 4) + ((int)threadIdx.y))] = red_buf0[0];
}



 __global__ void __launch_bounds__(128) cutlass_kernel_fp16_int4_fp16_m1n15360k5120_nt_4x8(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ C) {
            int8_t* B = ((int8_t *)QB);
	 half* Scales = (half *)((int8_t *)QB + 39321600);
	 int8_t* Zeros = ((int8_t *)QB + 39321600 + 1228800);                 
            // const dim3 GridDim(3840, 1, 1);
            // const dim3 BlockDim(32, 4, 1);
            // cutlass_kernel_fp16_int4_fp16_m1n15360k5120_nt_4x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
  half in_thread_C_local[1];
  half A_local[8];
  signed char Zeros_local[1];
  half red_buf0[1];
  in_thread_C_local[0] = __float2half_rn(0.000000e+00f);
  for (int k_0 = 0; k_0 < 20; ++k_0) {
    *(uint4*)(A_local + 0) = *(uint4*)(A + ((k_0 * 256) + (((int)threadIdx.x) * 8)));
    Zeros_local[0] = Zeros[(((k_0 * 15360) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.y))];
    for (int k_2 = 0; k_2 < 8; ++k_2) {
      in_thread_C_local[0] = (in_thread_C_local[0] + (A_local[k_2] * (((half)(((B[(((((((int)blockIdx.x) * 10240) + (((int)threadIdx.y) * 2560)) + (k_0 * 128)) + (((int)threadIdx.x) * 4)) + (k_2 >> 1))] >> ((signed char)((k_2 & 1) * 4))) & (signed char)15) - ((Zeros_local[0] >> ((signed char)((k_2 & 1) * 4))) & (signed char)15))) * Scales[((((k_0 * 30720) + ((((int)threadIdx.x) >> 4) * 15360)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.y))])));
    }
  }
  uint mask[1];
  half t0[1];
  red_buf0[0] = in_thread_C_local[0];
  mask[0] = (__activemask() & ((uint)(0 << (((int)threadIdx.y) * 32))));
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 16, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 8, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf0[0] = __shfl_sync(mask[0], red_buf0[0], (((int)threadIdx.y) * 32), 32);
  C[((((int)blockIdx.x) * 4) + ((int)threadIdx.y))] = red_buf0[0];
}



 __global__ void __launch_bounds__(128) cutlass_kernel_fp16_int4_fp16_m1n5120k5120_nt_4x8(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ C) {
            int8_t* B = ((int8_t *)QB);
	 half* Scales = (half *)((int8_t *)QB + 13107200);
	 int8_t* Zeros = ((int8_t *)QB + 13107200 + 409600);                 
            // const dim3 GridDim(1280, 1, 1);
            // const dim3 BlockDim(32, 4, 1);
            // cutlass_kernel_fp16_int4_fp16_m1n5120k5120_nt_4x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
  half in_thread_C_local[1];
  half A_local[8];
  signed char Zeros_local[1];
  half red_buf0[1];
  in_thread_C_local[0] = __float2half_rn(0.000000e+00f);
  for (int k_0 = 0; k_0 < 20; ++k_0) {
    *(uint4*)(A_local + 0) = *(uint4*)(A + ((k_0 * 256) + (((int)threadIdx.x) * 8)));
    Zeros_local[0] = Zeros[(((k_0 * 5120) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.y))];
    for (int k_2 = 0; k_2 < 8; ++k_2) {
      in_thread_C_local[0] = (in_thread_C_local[0] + (A_local[k_2] * (((half)(((B[(((((((int)blockIdx.x) * 10240) + (((int)threadIdx.y) * 2560)) + (k_0 * 128)) + (((int)threadIdx.x) * 4)) + (k_2 >> 1))] >> ((signed char)((k_2 & 1) * 4))) & (signed char)15) - ((Zeros_local[0] >> ((signed char)((k_2 & 1) * 4))) & (signed char)15))) * Scales[((((k_0 * 10240) + ((((int)threadIdx.x) >> 4) * 5120)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.y))])));
    }
  }
  uint mask[1];
  half t0[1];
  red_buf0[0] = in_thread_C_local[0];
  mask[0] = (__activemask() & ((uint)(0 << (((int)threadIdx.y) * 32))));
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 16, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 8, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf0[0] = __shfl_sync(mask[0], red_buf0[0], (((int)threadIdx.y) * 32), 32);
  C[((((int)blockIdx.x) * 4) + ((int)threadIdx.y))] = red_buf0[0];
}



 __global__ void __launch_bounds__(128) cutlass_kernel_fp16_int4_fp16_m1n5120k13824_nt_4x8(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ C) {
            int8_t* B = ((int8_t *)QB);
	 half* Scales = (half *)((int8_t *)QB + 35389440);
	 int8_t* Zeros = ((int8_t *)QB + 35389440 + 1105920);                 
            // const dim3 GridDim(1280, 1, 1);
            // const dim3 BlockDim(32, 4, 1);
            // cutlass_kernel_fp16_int4_fp16_m1n5120k13824_nt_4x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
  half in_thread_C_local[1];
  half A_local[8];
  signed char Zeros_local[1];
  half red_buf0[1];
  in_thread_C_local[0] = __float2half_rn(0.000000e+00f);
  for (int k_0 = 0; k_0 < 54; ++k_0) {
    *(uint4*)(A_local + 0) = *(uint4*)(A + ((k_0 * 256) + (((int)threadIdx.x) * 8)));
    if (k_0 < 52) {
      Zeros_local[0] = Zeros[(((k_0 * 5120) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.y))];
    }
    for (int k_2 = 0; k_2 < 8; ++k_2) {
      in_thread_C_local[0] = (in_thread_C_local[0] + (A_local[k_2] * (((half)(((B[(((((((int)blockIdx.x) * 27648) + (((int)threadIdx.y) * 6912)) + (k_0 * 128)) + (((int)threadIdx.x) * 4)) + (k_2 >> 1))] >> ((signed char)((k_2 & 1) * 4))) & (signed char)15) - ((Zeros_local[0] >> ((signed char)((k_2 & 1) * 4))) & (signed char)15))) * Scales[((((k_0 * 10240) + ((((int)threadIdx.x) >> 4) * 5120)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.y))])));
    }
  }
  uint mask[1];
  half t0[1];
  red_buf0[0] = in_thread_C_local[0];
  mask[0] = (__activemask() & ((uint)(0 << (((int)threadIdx.y) * 32))));
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 16, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 8, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf0[0] = __shfl_sync(mask[0], red_buf0[0], (((int)threadIdx.y) * 32), 32);
  C[((((int)blockIdx.x) * 4) + ((int)threadIdx.y))] = red_buf0[0];
}



 __global__ void __launch_bounds__(128) cutlass_kernel_fp16_int4_fp16_m1n13824k5120_nt_4x8(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ C) {
            int8_t* B = ((int8_t *)QB);
	 half* Scales = (half *)((int8_t *)QB + 35389440);
	 int8_t* Zeros = ((int8_t *)QB + 35389440 + 1105920);                 
            // const dim3 GridDim(3456, 1, 1);
            // const dim3 BlockDim(32, 4, 1);
            // cutlass_kernel_fp16_int4_fp16_m1n13824k5120_nt_4x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
  half in_thread_C_local[1];
  half A_local[8];
  signed char Zeros_local[1];
  half red_buf0[1];
  in_thread_C_local[0] = __float2half_rn(0.000000e+00f);
  for (int k_0 = 0; k_0 < 20; ++k_0) {
    *(uint4*)(A_local + 0) = *(uint4*)(A + ((k_0 * 256) + (((int)threadIdx.x) * 8)));
    Zeros_local[0] = Zeros[(((k_0 * 13824) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.y))];
    for (int k_2 = 0; k_2 < 8; ++k_2) {
      in_thread_C_local[0] = (in_thread_C_local[0] + (A_local[k_2] * (((half)(((B[(((((((int)blockIdx.x) * 10240) + (((int)threadIdx.y) * 2560)) + (k_0 * 128)) + (((int)threadIdx.x) * 4)) + (k_2 >> 1))] >> ((signed char)((k_2 & 1) * 4))) & (signed char)15) - ((Zeros_local[0] >> ((signed char)((k_2 & 1) * 4))) & (signed char)15))) * Scales[((((k_0 * 27648) + ((((int)threadIdx.x) >> 4) * 13824)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.y))])));
    }
  }
  uint mask[1];
  half t0[1];
  red_buf0[0] = in_thread_C_local[0];
  mask[0] = (__activemask() & ((uint)(0 << (((int)threadIdx.y) * 32))));
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 16, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 8, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf0[0] = __shfl_sync(mask[0], red_buf0[0], (((int)threadIdx.y) * 32), 32);
  C[((((int)blockIdx.x) * 4) + ((int)threadIdx.y))] = red_buf0[0];
}





int ladder_gemm_fp16xnf4_fp16(half *input_0, half *input_1, half *output, const int M, const int N, const int K, const int trans_a, const int trans_b)
{
    assert(trans_a == 0 && trans_b == 1);
    
        if (M == 1 && N == 12288 && K == 4096){
            
             const dim3 GridDim(3072, 1, 1);
             const dim3 BlockDim(32, 4, 1);
             cutlass_kernel_fp16_int4_fp16_m1n12288k4096_nt_4x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
        if (M == 1 && N == 4096 && K == 4096){
            
             const dim3 GridDim(1024, 1, 1);
             const dim3 BlockDim(32, 4, 1);
             cutlass_kernel_fp16_int4_fp16_m1n4096k4096_nt_4x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
        if (M == 1 && N == 4096 && K == 11008){
            
             const dim3 GridDim(1024, 1, 1);
             const dim3 BlockDim(32, 4, 1);
             cutlass_kernel_fp16_int4_fp16_m1n4096k11008_nt_4x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
        if (M == 1 && N == 11008 && K == 4096){
            
             const dim3 GridDim(2752, 1, 1);
             const dim3 BlockDim(32, 4, 1);
             cutlass_kernel_fp16_int4_fp16_m1n11008k4096_nt_4x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
        if (M == 1 && N == 15360 && K == 5120){
            
             const dim3 GridDim(3840, 1, 1);
             const dim3 BlockDim(32, 4, 1);
             cutlass_kernel_fp16_int4_fp16_m1n15360k5120_nt_4x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
        if (M == 1 && N == 5120 && K == 5120){
            
             const dim3 GridDim(1280, 1, 1);
             const dim3 BlockDim(32, 4, 1);
             cutlass_kernel_fp16_int4_fp16_m1n5120k5120_nt_4x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
        if (M == 1 && N == 5120 && K == 13824){
            
             const dim3 GridDim(1280, 1, 1);
             const dim3 BlockDim(32, 4, 1);
             cutlass_kernel_fp16_int4_fp16_m1n5120k13824_nt_4x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
        if (M == 1 && N == 13824 && K == 5120){
            
             const dim3 GridDim(3456, 1, 1);
             const dim3 BlockDim(32, 4, 1);
             cutlass_kernel_fp16_int4_fp16_m1n13824k5120_nt_4x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
    return -1;
}