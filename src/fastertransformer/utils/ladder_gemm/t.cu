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

 __global__ void __launch_bounds__(32) cutlass_kernel_fp16_nf4_fp16_m1n4096k4096_nt_1x8(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ C) {
            int8_t* B = ((int8_t *)QB);
	 half* LUT = (half *)((int8_t *)QB + 8388608);
	 half* Scales = (half *)((int8_t *)QB + 8388608 + 32);                 
            // const dim3 GridDim(4096, 1, 1);
            // const dim3 BlockDim(32, 1, 1);
            // cutlass_kernel_fp16_nf4_fp16_m1n4096k4096_nt_1x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
  half in_thread_C_local[1];
  half A_local[8];
  signed char B_local[4];
  half red_buf0[1];
  in_thread_C_local[0] = __float2half_rn(0.000000e+00f);
  for (int k_0 = 0; k_0 < 16; ++k_0) {
    *(uint4*)(A_local + 0) = *(uint4*)(A + ((k_0 * 256) + (((int)threadIdx.x) * 8)));
    *(int*)(B_local + 0) = *(int*)(B + (((((int)blockIdx.x) * 2048) + (k_0 * 128)) + (((int)threadIdx.x) * 4)));
    for (int k_2 = 0; k_2 < 8; ++k_2) {
      in_thread_C_local[0] = (in_thread_C_local[0] + (A_local[k_2] * (LUT[((int)((B_local[(k_2 >> 1)] >> ((signed char)((k_2 & 1) * 4))) & (signed char)15))] * Scales[(((k_0 * 8192) + ((((int)threadIdx.x) >> 4) * 4096)) + ((int)blockIdx.x))])));
    }
  }
  uint mask[1];
  half t0[1];
  red_buf0[0] = in_thread_C_local[0];
  mask[0] = __activemask();
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
  red_buf0[0] = __shfl_sync(mask[0], red_buf0[0], 0, 32);
  C[((int)blockIdx.x)] = red_buf0[0];
}



 __global__ void __launch_bounds__(32) cutlass_kernel_fp16_nf4_fp16_m1n4096k11008_nt_1x8(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ C) {
            int8_t* B = ((int8_t *)QB);
	 half* LUT = (half *)((int8_t *)QB + 22544384);
	 half* Scales = (half *)((int8_t *)QB + 22544384 + 32);                 
            // const dim3 GridDim(4096, 1, 1);
            // const dim3 BlockDim(32, 1, 1);
            // cutlass_kernel_fp16_nf4_fp16_m1n4096k11008_nt_1x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
  half in_thread_C_local[1];
  half A_local[8];
  signed char B_local[4];
  half red_buf0[1];
  in_thread_C_local[0] = __float2half_rn(0.000000e+00f);
  for (int k_0 = 0; k_0 < 43; ++k_0) {
    *(uint4*)(A_local + 0) = *(uint4*)(A + ((k_0 * 256) + (((int)threadIdx.x) * 8)));
    *(int*)(B_local + 0) = *(int*)(B + (((((int)blockIdx.x) * 5504) + (k_0 * 128)) + (((int)threadIdx.x) * 4)));
    for (int k_2 = 0; k_2 < 8; ++k_2) {
      in_thread_C_local[0] = (in_thread_C_local[0] + (A_local[k_2] * (LUT[((int)((B_local[(k_2 >> 1)] >> ((signed char)((k_2 & 1) * 4))) & (signed char)15))] * Scales[(((k_0 * 8192) + ((((int)threadIdx.x) >> 4) * 4096)) + ((int)blockIdx.x))])));
    }
  }
  uint mask[1];
  half t0[1];
  red_buf0[0] = in_thread_C_local[0];
  mask[0] = __activemask();
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
  red_buf0[0] = __shfl_sync(mask[0], red_buf0[0], 0, 32);
  C[((int)blockIdx.x)] = red_buf0[0];
}



 __global__ void __launch_bounds__(32) cutlass_kernel_fp16_nf4_fp16_m1n11008k4096_nt_1x8(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ C) {
            int8_t* B = ((int8_t *)QB);
	 half* LUT = (half *)((int8_t *)QB + 22544384);
	 half* Scales = (half *)((int8_t *)QB + 22544384 + 32);                 
            // const dim3 GridDim(11008, 1, 1);
            // const dim3 BlockDim(32, 1, 1);
            // cutlass_kernel_fp16_nf4_fp16_m1n11008k4096_nt_1x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
  half in_thread_C_local[1];
  half A_local[8];
  signed char B_local[4];
  half red_buf0[1];
  in_thread_C_local[0] = __float2half_rn(0.000000e+00f);
  for (int k_0 = 0; k_0 < 16; ++k_0) {
    *(uint4*)(A_local + 0) = *(uint4*)(A + ((k_0 * 256) + (((int)threadIdx.x) * 8)));
    *(int*)(B_local + 0) = *(int*)(B + (((((int)blockIdx.x) * 2048) + (k_0 * 128)) + (((int)threadIdx.x) * 4)));
    for (int k_2 = 0; k_2 < 8; ++k_2) {
      in_thread_C_local[0] = (in_thread_C_local[0] + (A_local[k_2] * (LUT[((int)((B_local[(k_2 >> 1)] >> ((signed char)((k_2 & 1) * 4))) & (signed char)15))] * Scales[(((k_0 * 22016) + ((((int)threadIdx.x) >> 4) * 11008)) + ((int)blockIdx.x))])));
    }
  }
  uint mask[1];
  half t0[1];
  red_buf0[0] = in_thread_C_local[0];
  mask[0] = __activemask();
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
  red_buf0[0] = __shfl_sync(mask[0], red_buf0[0], 0, 32);
  C[((int)blockIdx.x)] = red_buf0[0];
}





int ladder_gemm_fp16xnf4_fp16(half *input_0, half *input_1, half *output, const int M, const int N, const int K, const int trans_a, const int trans_b)
{
    assert(trans_a == 0 && trans_b == 1);
    
        if (M == 1 && N == 4096 && K == 4096){
            
             const dim3 GridDim(4096, 1, 1);
             const dim3 BlockDim(32, 1, 1);
             cutlass_kernel_fp16_nf4_fp16_m1n4096k4096_nt_1x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
        if (M == 1 && N == 4096 && K == 11008){
            
             const dim3 GridDim(4096, 1, 1);
             const dim3 BlockDim(32, 1, 1);
             cutlass_kernel_fp16_nf4_fp16_m1n4096k11008_nt_1x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
        if (M == 1 && N == 11008 && K == 4096){
            
             const dim3 GridDim(11008, 1, 1);
             const dim3 BlockDim(32, 1, 1);
             cutlass_kernel_fp16_nf4_fp16_m1n11008k4096_nt_1x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
    return -1;
}