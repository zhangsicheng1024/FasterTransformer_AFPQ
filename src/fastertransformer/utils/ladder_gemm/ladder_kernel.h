#ifndef __LADDER_KERNEL_H__
#define __LADDER_KERNEL_H__

#include <cuda_runtime.h>
#include <cuda_fp16.h>

int ladder_gemm_fp16xnf4_fp16(half *input_0, half *input_1, half *output, const int M, const int N, const int K, const int trans_a, const int trans_b);

#endif
