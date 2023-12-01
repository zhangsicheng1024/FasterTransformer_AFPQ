#include <cuda_runtime.h>
#include <assert.h>
#include "ladder_kernel.h"
#include "mma.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/warp/mma_tensor_op.h"
#include "cutlass/gemm/warp/mma_tensor_op_sm70.h"

namespace cutlass {
namespace gemm {
namespace warp {

template<class MmaWarp, int KSize>
class MMAWarpWrapper {
public:
  typename MmaWarp::FragmentA frag_A[2];
  typename MmaWarp::FragmentB frag_B[2];
  typename MmaWarp::FragmentC accum;
  MmaWarp mma_op;
  typename MmaWarp::IteratorA iter_A;
  typename MmaWarp::IteratorB iter_B;
  const int warp_idx_m_, warp_idx_n_, lane_id_;

  using TensorRefA = typename MmaWarp::IteratorA::TensorRef;
  using TensorRefB = typename MmaWarp::IteratorB::TensorRef;
  static_assert(KSize % MmaWarp::Shape::kK == 0);
  static int constexpr kKgroups = KSize / MmaWarp::Shape::kK;

  CUTLASS_DEVICE
  MMAWarpWrapper(int warp_idx_m, int warp_idx_n, int lane_id)
  : warp_idx_m_(warp_idx_m), warp_idx_n_(warp_idx_n), lane_id_(lane_id), iter_A({nullptr, 0}, 0), iter_B({nullptr, 0}, 0) {
    accum.clear();
  }

  CUTLASS_DEVICE
  void prologue(const TensorRefA &ref_A, const TensorRefB &ref_B) {
    iter_A = typename MmaWarp::IteratorA(ref_A, lane_id_);
    iter_B = typename MmaWarp::IteratorB(ref_B, lane_id_);
    iter_A.add_tile_offset({warp_idx_m_, 0});
    iter_B.add_tile_offset({0, warp_idx_n_});
    iter_A.load(frag_A[0]);
    iter_B.load(frag_B[0]);
    ++iter_A;
    ++iter_B;
  }
  CUTLASS_DEVICE
  void body() {
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < kKgroups - 1; ++k) {
      iter_A.load(frag_A[(k + 1) % 2]);
      iter_B.load(frag_B[(k + 1) % 2]);
      ++iter_A;
      ++iter_B;
      mma_op(accum, frag_A[k % 2], frag_B[k % 2], accum);
    }
    __syncthreads();
  }
  CUTLASS_DEVICE
  void epilogue() {
    mma_op(accum, frag_A[(kKgroups - 1) % 2], frag_B[(kKgroups - 1) % 2], accum);
  }
};

template <
  typename Shape,
  typename SMemLayoutA,
  typename SMemLayoutB
>
class GemmTensorOp {
public:
  using InstructionShape = GemmShape<16, 8, 16>;
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      InstructionShape,
      32,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::layout::ColumnMajor,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using MmaWarp = typename cutlass::gemm::warp::MmaTensorOp<
    GemmShape<Shape::kM, Shape::kN, InstructionShape::kK>,
    cutlass::half_t,
    SMemLayoutA,
    cutlass::half_t,
    SMemLayoutB,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    Policy
  >;
  using MMA = MMAWarpWrapper<MmaWarp, Shape::kK>;
  MMA mma;

  CUTLASS_DEVICE
  GemmTensorOp(int warp_idx_m, int warp_idx_n, int lane_id)
  : mma(warp_idx_m, warp_idx_n, lane_id) {}
  CUTLASS_DEVICE
  half& operator[](size_t i) const {
    return ((half*)mma.accum.data())[i];
  }
  CUTLASS_DEVICE
  half* operator+(size_t i) const {
    return (half*)mma.accum.data() + i;
  }
};

template <
  typename Shape,
  typename SMemLayoutA,
  typename LayoutA,
  typename SMemLayoutB,
  typename LayoutB,
  typename LayoutC
>
class VoltaGemmTensorOp {
public:
  using InstructionShape = GemmShape<16, 16, 4>;
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      InstructionShape,
      32,
      cutlass::half_t,
      LayoutA,
      cutlass::half_t,
      LayoutB,
      cutlass::half_t,
      LayoutC,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using MmaWarp = typename cutlass::gemm::warp::MmaVoltaTensorOp<
    GemmShape<Shape::kM, Shape::kN, InstructionShape::kK>,
    cutlass::half_t,
    SMemLayoutA,
    cutlass::half_t,
    SMemLayoutB,
    cutlass::half_t,
    LayoutC,
    Policy
  >;
  using MMA = MMAWarpWrapper<MmaWarp, Shape::kK>;
  MMA mma;

  CUTLASS_DEVICE
  VoltaGemmTensorOp(int warp_idx_m, int warp_idx_n, int lane_id)
  : mma(warp_idx_m, warp_idx_n, lane_id) {}
  CUTLASS_DEVICE
  half& operator[](size_t i) const {
    return ((half*)mma.accum.data())[i];
  }
  CUTLASS_DEVICE
  half* operator+(size_t i) const {
    return (half*)mma.accum.data() + i;
  }
};

template <
  typename Shape,
  typename SMemLayoutA,
  typename SMemLayoutB
>
class GemmI8TensorOp {
public:
  using InstructionShape = GemmShape<16, 8, 32>;
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      InstructionShape,
      32,
      int8_t,
      cutlass::layout::RowMajor,
      int8_t,
      cutlass::layout::ColumnMajor,
      int,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using MmaWarp = typename cutlass::gemm::warp::MmaTensorOp<
    GemmShape<Shape::kM, Shape::kN, InstructionShape::kK>,
    int8_t,
    SMemLayoutA,
    int8_t,
    SMemLayoutB,
    int,
    cutlass::layout::RowMajor,
    Policy
  >;
  using MMA = MMAWarpWrapper<MmaWarp, Shape::kK>;
  MMA mma;

  CUTLASS_DEVICE
  GemmI8TensorOp(int warp_idx_m, int warp_idx_n, int lane_id)
  : mma(warp_idx_m, warp_idx_n, lane_id) {}
  CUTLASS_DEVICE
  int& operator[](size_t i) const {
    return ((int*)mma.accum.data())[i];
  }
  CUTLASS_DEVICE
  int* operator+(size_t i) const {
    return (int*)mma.accum.data() + i;
  }
};

}}}

template<class TensorOp>
CUTLASS_DEVICE void call_cutlass_mma_body(TensorOp& op) {
  op.mma.body();
}

template<class TensorOp>
CUTLASS_DEVICE void call_cutlass_mma_epilogue(TensorOp& op) {
  op.mma.epilogue();
}

template<class TensorOp>
CUTLASS_DEVICE void call_cutlass_mma_prologue(TensorOp& op, void* pA, void* pB, int sA, int sB) {
  using TensorRefA = typename TensorOp::MMA::TensorRefA;
  using TensorRefB = typename TensorOp::MMA::TensorRefB;
  TensorRefA refA{(typename TensorRefA::Element*)pA, sA};
  TensorRefB refB{(typename TensorRefB::Element*)pB, sB};
  op.mma.prologue(refA, refB);
}

#define ALLOCATE_CUTLASS_OBJECT(var, ...) auto var = __VA_ARGS__;

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y)
{
    unsigned v0 = *((unsigned short *)&x);
    unsigned v1 = *((unsigned short *)&y);
    return (v1 << 16) | v0;
}

__global__ void __launch_bounds__(96) cutlass_kernel_fp16_nf4_fp16_m16n12288k4096_nt_16x48x32_16x8x16_(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ C) {
            int8_t* B = ((int8_t *)QB);
	 half* Scales = (half *)((int8_t *)QB + 25165824);                 
            // const dim3 GridDim(256, 1, 1);
            // const dim3 BlockDim(32, 3, 1);
            // cutlass_kernel_fp16_nf4_fp16_m16n12288k4096_nt_16x48x32_16x8x16_<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
  
    __shared__ half A_shared[1024];
  __shared__ signed char B_shared[1536];
  __shared__ half B_decode_shared[1536];
  half B_decode_local[8];
  signed char B_local[2];
  half B_decode_local_1[8];
  signed char B_local_1[2];
  ALLOCATE_CUTLASS_OBJECT(C_cutlass_warp_mma, cutlass::gemm::warp::GemmTensorOp<
    cutlass::gemm::GemmShape<16, 16, 32>,
    cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<16, 32>,
    cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<16, 32>
>(0, ((int)threadIdx.y), ((int)threadIdx.x)));
  #pragma unroll
  for (int ax0_ax1_fused_0_0_0 = 0; ax0_ax1_fused_0_0_0 < 1; ++ax0_ax1_fused_0_0_0) {
    if (((int)threadIdx.y) < 2) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + ((((((int)threadIdx.y) * 256) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(A_shared + ((((((int)threadIdx.y) * 256) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + (((((int)threadIdx.y) * 32768) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)))), "n"(16)
    );
  }
    }
  }
  for (int ax0_ax1_0_fused_0_0 = 0; ax0_ax1_0_fused_0_0 < 2; ++ax0_ax1_0_fused_0_0) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(B_shared + (((ax0_ax1_0_fused_0_0 * 384) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(B_shared + (((ax0_ax1_0_fused_0_0 * 384) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.ca.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + (((((((int)blockIdx.x) * 98304) + (ax0_ax1_0_fused_0_0 * 49152)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 2) * 2048)) + ((((int)threadIdx.x) & 3) * 4)))), "n"(4)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

  for (int k_0 = 0; k_0 < 127; ++k_0) {
    #pragma unroll
    for (int ax0_ax1_fused_0_0_0_1 = 0; ax0_ax1_fused_0_0_0_1 < 1; ++ax0_ax1_fused_0_0_0_1) {
      if (((int)threadIdx.y) < 2) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + (((((((k_0 + 1) & 1) * 512) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(A_shared + (((((((k_0 + 1) & 1) * 512) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + (((((((int)threadIdx.y) * 32768) + ((((int)threadIdx.x) >> 2) * 4096)) + (k_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 32))), "n"(16)
    );
  }
      }
    }
    for (int ax0_ax1_0_fused_0_0_1 = 0; ax0_ax1_0_fused_0_0_1 < 2; ++ax0_ax1_0_fused_0_0_1) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(B_shared + ((((((k_0 + 1) & 1) * 768) + (ax0_ax1_0_fused_0_0_1 * 384)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(B_shared + ((((((k_0 + 1) & 1) * 768) + (ax0_ax1_0_fused_0_0_1 * 384)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.ca.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + (((((((((int)blockIdx.x) * 98304) + (ax0_ax1_0_fused_0_0_1 * 49152)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 2) * 2048)) + (k_0 * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 16))), "n"(4)
    );
  }
    }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 1;");

    __syncthreads();
    for (int ax0_ax1_0_fused_0_0_2 = 0; ax0_ax1_0_fused_0_0_2 < 2; ++ax0_ax1_0_fused_0_0_2) {
      for (int ax0_0 = 0; ax0_0 < 2; ++ax0_0) {
        *(char2*)(B_local + 0) = *(char2*)(B_shared + ((((((k_0 & 1) * 768) + (ax0_ax1_0_fused_0_0_2 * 384)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + (ax0_0 * 2)));
        for (int ax0_1 = 0; ax0_1 < 4; ++ax0_1) {
          B_decode_local[((ax0_0 * 4) + ax0_1)] = ((((half)((B_local[(ax0_1 >> 1)] >> ((signed char)((ax0_1 & 1) * 4))) & (signed char)15)) - __float2half_rn(8.000000e+00f)) * Scales[((((((((((k_0 * 32) + ((((int)threadIdx.x) & 3) * 8)) + (ax0_0 * 4)) + ax0_1) >> 7) * 12288) + (((int)blockIdx.x) * 48)) + (ax0_ax1_0_fused_0_0_2 * 24)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))]);
        }
      }
      *(uint4*)(B_decode_shared + (((((ax0_ax1_0_fused_0_0_2 * 768) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(B_decode_local + 0);
    }
    __syncthreads();
    call_cutlass_mma_prologue(C_cutlass_warp_mma, (&(A_shared[((k_0 & 1) * 512)])), (&(B_decode_shared[0])), 32, 32);
    call_cutlass_mma_body(C_cutlass_warp_mma);
    call_cutlass_mma_epilogue(C_cutlass_warp_mma);
  }
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  for (int ax0_ax1_0_fused_0_0_3 = 0; ax0_ax1_0_fused_0_0_3 < 2; ++ax0_ax1_0_fused_0_0_3) {
    for (int ax0_0_1 = 0; ax0_0_1 < 2; ++ax0_0_1) {
      *(char2*)(B_local_1 + 0) = *(char2*)(B_shared + (((((ax0_ax1_0_fused_0_0_3 * 384) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + (ax0_0_1 * 2)) + 768));
      for (int ax0_1_1 = 0; ax0_1_1 < 4; ++ax0_1_1) {
        B_decode_local_1[((ax0_0_1 * 4) + ax0_1_1)] = ((((half)((B_local_1[(ax0_1_1 >> 1)] >> ((signed char)((ax0_1_1 & 1) * 4))) & (signed char)15)) - __float2half_rn(8.000000e+00f)) * Scales[(((((((((((((int)threadIdx.x) & 3) * 8) + (ax0_0_1 * 4)) + ax0_1_1) + 4064) >> 7) * 12288) + (((int)blockIdx.x) * 48)) + (ax0_ax1_0_fused_0_0_3 * 24)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))]);
      }
    }
    *(uint4*)(B_decode_shared + (((((ax0_ax1_0_fused_0_0_3 * 768) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(B_decode_local_1 + 0);
  }
  __syncthreads();
  call_cutlass_mma_prologue(C_cutlass_warp_mma, (&(A_shared[512])), (&(B_decode_shared[0])), 32, 32);
  call_cutlass_mma_body(C_cutlass_warp_mma);
  call_cutlass_mma_epilogue(C_cutlass_warp_mma);
  #pragma unroll
  for (int ax1_0 = 0; ax1_0 < 4; ++ax1_0) {
    *(uint1*)(C + (((((((ax1_0 & 1) * 98304) + ((((int)threadIdx.x) >> 2) * 12288)) + (((int)blockIdx.x) * 48)) + (((int)threadIdx.y) * 16)) + ((ax1_0 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(C_cutlass_warp_mma + (ax1_0 * 2));
  }
}



__global__ void __launch_bounds__(64) cutlass_kernel_fp16_nf4_fp16_m16n4096k4096_nt_16x16x32_16x8x16_(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ C) {
            int8_t* B = ((int8_t *)QB);
	 half* Scales = (half *)((int8_t *)QB + 8388608);                 
            // const dim3 GridDim(256, 1, 1);
            // const dim3 BlockDim(32, 2, 1);
            // cutlass_kernel_fp16_nf4_fp16_m16n4096k4096_nt_16x16x32_16x8x16_<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
  
    __shared__ half A_shared[1024];
  __shared__ signed char B_shared[1280];
  __shared__ half B_decode_shared[640];
  half B_decode_local[8];
  signed char B_local[2];
  half B_decode_local_1[8];
  signed char B_local_1[2];
  ALLOCATE_CUTLASS_OBJECT(C_cutlass_warp_mma, cutlass::gemm::warp::GemmTensorOp<
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<16, 32>,
    cutlass::layout::ColumnMajor
>(0, ((int)threadIdx.y), ((int)threadIdx.x)));
  #pragma unroll
  for (int ax0_ax1_fused_0_0_0 = 0; ax0_ax1_fused_0_0_0 < 1; ++ax0_ax1_fused_0_0_0) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + ((((((int)threadIdx.y) * 256) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(A_shared + ((((((int)threadIdx.y) * 256) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + (((((int)threadIdx.y) * 32768) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)))), "n"(16)
    );
  }
  }

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(B_shared + (((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 4)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(B_shared + (((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 4))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.ca.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 2) * 2048)) + ((((int)threadIdx.x) & 3) * 4)))), "n"(4)
    );
  }
__asm__ __volatile__("cp.async.commit_group;");

  for (int k_0 = 0; k_0 < 127; ++k_0) {
    #pragma unroll
    for (int ax0_ax1_fused_0_0_0_1 = 0; ax0_ax1_fused_0_0_0_1 < 1; ++ax0_ax1_fused_0_0_0_1) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + (((((((k_0 + 1) & 1) * 512) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(A_shared + (((((((k_0 + 1) & 1) * 512) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + (((((((int)threadIdx.y) * 32768) + ((((int)threadIdx.x) >> 2) * 4096)) + (k_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 32))), "n"(16)
    );
  }
    }

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(B_shared + ((((((k_0 + 1) & 1) * 640) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 4)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(B_shared + ((((((k_0 + 1) & 1) * 640) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 4))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.ca.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((int)blockIdx.x) * 32768) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 2) * 2048)) + (k_0 * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 16))), "n"(4)
    );
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 1;");

    __syncthreads();
    for (int ax0_0 = 0; ax0_0 < 2; ++ax0_0) {
      *(char2*)(B_local + 0) = *(char2*)(B_shared + ((((((k_0 & 1) * 640) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 4)) + (ax0_0 * 2)));
      for (int ax0_1 = 0; ax0_1 < 4; ++ax0_1) {
        B_decode_local[((ax0_0 * 4) + ax0_1)] = ((((half)((B_local[(ax0_1 >> 1)] >> ((signed char)((ax0_1 & 1) * 4))) & (signed char)15)) - __float2half_rn(8.000000e+00f)) * Scales[(((((((((k_0 * 32) + ((((int)threadIdx.x) & 3) * 8)) + (ax0_0 * 4)) + ax0_1) >> 7) * 4096) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))]);
      }
    }
    *(uint4*)(B_decode_shared + (((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) = *(uint4*)(B_decode_local + 0);
    __syncthreads();
    call_cutlass_mma_prologue(C_cutlass_warp_mma, (&(A_shared[((k_0 & 1) * 512)])), (&(B_decode_shared[0])), 32, 40);
    call_cutlass_mma_body(C_cutlass_warp_mma);
    call_cutlass_mma_epilogue(C_cutlass_warp_mma);
  }
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  for (int ax0_0_1 = 0; ax0_0_1 < 2; ++ax0_0_1) {
    *(char2*)(B_local_1 + 0) = *(char2*)(B_shared + (((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 4)) + (ax0_0_1 * 2)) + 640));
    for (int ax0_1_1 = 0; ax0_1_1 < 4; ++ax0_1_1) {
      B_decode_local_1[((ax0_0_1 * 4) + ax0_1_1)] = ((((half)((B_local_1[(ax0_1_1 >> 1)] >> ((signed char)((ax0_1_1 & 1) * 4))) & (signed char)15)) - __float2half_rn(8.000000e+00f)) * Scales[((((((((((((int)threadIdx.x) & 3) * 8) + (ax0_0_1 * 4)) + ax0_1_1) + 4064) >> 7) * 4096) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))]);
    }
  }
  *(uint4*)(B_decode_shared + (((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) = *(uint4*)(B_decode_local_1 + 0);
  __syncthreads();
  call_cutlass_mma_prologue(C_cutlass_warp_mma, (&(A_shared[512])), (&(B_decode_shared[0])), 32, 40);
  call_cutlass_mma_body(C_cutlass_warp_mma);
  call_cutlass_mma_epilogue(C_cutlass_warp_mma);
  #pragma unroll
  for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0) {
    *(uint1*)(C + (((((ax1_0 * 32768) + ((((int)threadIdx.x) >> 2) * 4096)) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.y) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(C_cutlass_warp_mma + (ax1_0 * 2));
  }
}



__global__ void __launch_bounds__(128) cutlass_kernel_fp16_nf4_fp16_m16n11008k4096_nt_16x64x32_16x8x16_(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ C) {
            int8_t* B = ((int8_t *)QB);
	 half* Scales = (half *)((int8_t *)QB + 22544384);                 
            // const dim3 GridDim(172, 1, 1);
            // const dim3 BlockDim(32, 4, 1);
            // cutlass_kernel_fp16_nf4_fp16_m16n11008k4096_nt_16x64x32_16x8x16_<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
  
    __shared__ half A_shared[1024];
  __shared__ signed char B_shared[2048];
  __shared__ half B_decode_shared[2048];
  half B_decode_local[8];
  signed char B_local[2];
  half B_decode_local_1[8];
  signed char B_local_1[2];
  ALLOCATE_CUTLASS_OBJECT(C_cutlass_warp_mma, cutlass::gemm::warp::GemmTensorOp<
    cutlass::gemm::GemmShape<16, 16, 32>,
    cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<16, 32>,
    cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<16, 32>
>(0, ((int)threadIdx.y), ((int)threadIdx.x)));
  #pragma unroll
  for (int ax0_ax1_fused_0_0_0 = 0; ax0_ax1_fused_0_0_0 < 1; ++ax0_ax1_fused_0_0_0) {
    if (((int)threadIdx.y) < 2) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + ((((((int)threadIdx.y) * 256) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(A_shared + ((((((int)threadIdx.y) * 256) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + (((((int)threadIdx.y) * 32768) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)))), "n"(16)
    );
  }
    }
  }
  for (int ax0_ax1_0_fused_0_0 = 0; ax0_ax1_0_fused_0_0 < 2; ++ax0_ax1_0_fused_0_0) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(B_shared + (((ax0_ax1_0_fused_0_0 * 512) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(B_shared + (((ax0_ax1_0_fused_0_0 * 512) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.ca.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + (((((((int)blockIdx.x) * 131072) + (ax0_ax1_0_fused_0_0 * 65536)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 2) * 2048)) + ((((int)threadIdx.x) & 3) * 4)))), "n"(4)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

  for (int k_0 = 0; k_0 < 127; ++k_0) {
    #pragma unroll
    for (int ax0_ax1_fused_0_0_0_1 = 0; ax0_ax1_fused_0_0_0_1 < 1; ++ax0_ax1_fused_0_0_0_1) {
      if (((int)threadIdx.y) < 2) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + (((((((k_0 + 1) & 1) * 512) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(A_shared + (((((((k_0 + 1) & 1) * 512) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + (((((((int)threadIdx.y) * 32768) + ((((int)threadIdx.x) >> 2) * 4096)) + (k_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 32))), "n"(16)
    );
  }
      }
    }
    for (int ax0_ax1_0_fused_0_0_1 = 0; ax0_ax1_0_fused_0_0_1 < 2; ++ax0_ax1_0_fused_0_0_1) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(B_shared + ((((((k_0 + 1) & 1) * 1024) + (ax0_ax1_0_fused_0_0_1 * 512)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(B_shared + ((((((k_0 + 1) & 1) * 1024) + (ax0_ax1_0_fused_0_0_1 * 512)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.ca.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + (((((((((int)blockIdx.x) * 131072) + (ax0_ax1_0_fused_0_0_1 * 65536)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 2) * 2048)) + (k_0 * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 16))), "n"(4)
    );
  }
    }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 1;");

    __syncthreads();
    for (int ax0_ax1_0_fused_0_0_2 = 0; ax0_ax1_0_fused_0_0_2 < 2; ++ax0_ax1_0_fused_0_0_2) {
      for (int ax0_0 = 0; ax0_0 < 2; ++ax0_0) {
        *(char2*)(B_local + 0) = *(char2*)(B_shared + ((((((k_0 & 1) * 1024) + (ax0_ax1_0_fused_0_0_2 * 512)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + (ax0_0 * 2)));
        for (int ax0_1 = 0; ax0_1 < 4; ++ax0_1) {
          B_decode_local[((ax0_0 * 4) + ax0_1)] = ((((half)((B_local[(ax0_1 >> 1)] >> ((signed char)((ax0_1 & 1) * 4))) & (signed char)15)) - __float2half_rn(8.000000e+00f)) * Scales[((((((((((k_0 * 32) + ((((int)threadIdx.x) & 3) * 8)) + (ax0_0 * 4)) + ax0_1) >> 7) * 11008) + (((int)blockIdx.x) * 64)) + (ax0_ax1_0_fused_0_0_2 * 32)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))]);
        }
      }
      *(uint4*)(B_decode_shared + (((((ax0_ax1_0_fused_0_0_2 * 1024) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(B_decode_local + 0);
    }
    __syncthreads();
    call_cutlass_mma_prologue(C_cutlass_warp_mma, (&(A_shared[((k_0 & 1) * 512)])), (&(B_decode_shared[0])), 32, 32);
    call_cutlass_mma_body(C_cutlass_warp_mma);
    call_cutlass_mma_epilogue(C_cutlass_warp_mma);
  }
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  for (int ax0_ax1_0_fused_0_0_3 = 0; ax0_ax1_0_fused_0_0_3 < 2; ++ax0_ax1_0_fused_0_0_3) {
    for (int ax0_0_1 = 0; ax0_0_1 < 2; ++ax0_0_1) {
      *(char2*)(B_local_1 + 0) = *(char2*)(B_shared + (((((ax0_ax1_0_fused_0_0_3 * 512) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + (ax0_0_1 * 2)) + 1024));
      for (int ax0_1_1 = 0; ax0_1_1 < 4; ++ax0_1_1) {
        B_decode_local_1[((ax0_0_1 * 4) + ax0_1_1)] = ((((half)((B_local_1[(ax0_1_1 >> 1)] >> ((signed char)((ax0_1_1 & 1) * 4))) & (signed char)15)) - __float2half_rn(8.000000e+00f)) * Scales[(((((((((((((int)threadIdx.x) & 3) * 8) + (ax0_0_1 * 4)) + ax0_1_1) + 4064) >> 7) * 11008) + (((int)blockIdx.x) * 64)) + (ax0_ax1_0_fused_0_0_3 * 32)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))]);
      }
    }
    *(uint4*)(B_decode_shared + (((((ax0_ax1_0_fused_0_0_3 * 1024) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(B_decode_local_1 + 0);
  }
  __syncthreads();
  call_cutlass_mma_prologue(C_cutlass_warp_mma, (&(A_shared[512])), (&(B_decode_shared[0])), 32, 32);
  call_cutlass_mma_body(C_cutlass_warp_mma);
  call_cutlass_mma_epilogue(C_cutlass_warp_mma);
  #pragma unroll
  for (int ax1_0 = 0; ax1_0 < 4; ++ax1_0) {
    *(uint1*)(C + (((((((ax1_0 & 1) * 88064) + ((((int)threadIdx.x) >> 2) * 11008)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 16)) + ((ax1_0 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(C_cutlass_warp_mma + (ax1_0 * 2));
  }
}



__global__ void __launch_bounds__(128) cutlass_kernel_fp16_nf4_fp16_m16n4096k11008_nt_16x128x32_16x8x16_(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ C) {
            int8_t* B = ((int8_t *)QB);
	 half* Scales = (half *)((int8_t *)QB + 22544384);                 
            // const dim3 GridDim(32, 1, 1);
            // const dim3 BlockDim(32, 4, 1);
            // cutlass_kernel_fp16_nf4_fp16_m16n4096k11008_nt_16x128x32_16x8x16_<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
  
    __shared__ half A_shared[1024];
  __shared__ signed char B_shared[4096];
  __shared__ half B_decode_shared[4096];
  half B_decode_local[8];
  signed char B_local[2];
  half B_decode_local_1[8];
  signed char B_local_1[2];
  ALLOCATE_CUTLASS_OBJECT(C_cutlass_warp_mma, cutlass::gemm::warp::GemmTensorOp<
    cutlass::gemm::GemmShape<16, 32, 32>,
    cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<16, 32>,
    cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<16, 32>
>(0, ((int)threadIdx.y), ((int)threadIdx.x)));
  #pragma unroll
  for (int ax0_ax1_fused_0_0_0 = 0; ax0_ax1_fused_0_0_0 < 1; ++ax0_ax1_fused_0_0_0) {
    if (((int)threadIdx.y) < 2) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + ((((((int)threadIdx.y) * 256) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(A_shared + ((((((int)threadIdx.y) * 256) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + (((((int)threadIdx.y) * 88064) + ((((int)threadIdx.x) >> 2) * 11008)) + ((((int)threadIdx.x) & 3) * 8)))), "n"(16)
    );
  }
    }
  }
  for (int ax0_ax1_0_fused_0_0 = 0; ax0_ax1_0_fused_0_0 < 4; ++ax0_ax1_0_fused_0_0) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(B_shared + (((ax0_ax1_0_fused_0_0 * 512) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(B_shared + (((ax0_ax1_0_fused_0_0 * 512) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.ca.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + (((((((int)blockIdx.x) * 704512) + (ax0_ax1_0_fused_0_0 * 176128)) + (((int)threadIdx.y) * 44032)) + ((((int)threadIdx.x) >> 2) * 5504)) + ((((int)threadIdx.x) & 3) * 4)))), "n"(4)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

  for (int k_0 = 0; k_0 < 343; ++k_0) {
    #pragma unroll
    for (int ax0_ax1_fused_0_0_0_1 = 0; ax0_ax1_fused_0_0_0_1 < 1; ++ax0_ax1_fused_0_0_0_1) {
      if (((int)threadIdx.y) < 2) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + (((((((k_0 + 1) & 1) * 512) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(A_shared + (((((((k_0 + 1) & 1) * 512) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + (((((((int)threadIdx.y) * 88064) + ((((int)threadIdx.x) >> 2) * 11008)) + (k_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 32))), "n"(16)
    );
  }
      }
    }
    for (int ax0_ax1_0_fused_0_0_1 = 0; ax0_ax1_0_fused_0_0_1 < 4; ++ax0_ax1_0_fused_0_0_1) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(B_shared + ((((((k_0 + 1) & 1) * 2048) + (ax0_ax1_0_fused_0_0_1 * 512)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(B_shared + ((((((k_0 + 1) & 1) * 2048) + (ax0_ax1_0_fused_0_0_1 * 512)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.ca.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + (((((((((int)blockIdx.x) * 704512) + (ax0_ax1_0_fused_0_0_1 * 176128)) + (((int)threadIdx.y) * 44032)) + ((((int)threadIdx.x) >> 2) * 5504)) + (k_0 * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 16))), "n"(4)
    );
  }
    }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 1;");

    __syncthreads();
    for (int ax0_ax1_0_fused_0_0_2 = 0; ax0_ax1_0_fused_0_0_2 < 4; ++ax0_ax1_0_fused_0_0_2) {
      for (int ax0_0 = 0; ax0_0 < 2; ++ax0_0) {
        *(char2*)(B_local + 0) = *(char2*)(B_shared + ((((((k_0 & 1) * 2048) + (ax0_ax1_0_fused_0_0_2 * 512)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + (ax0_0 * 2)));
        for (int ax0_1 = 0; ax0_1 < 4; ++ax0_1) {
          B_decode_local[((ax0_0 * 4) + ax0_1)] = ((((half)((B_local[(ax0_1 >> 1)] >> ((signed char)((ax0_1 & 1) * 4))) & (signed char)15)) - __float2half_rn(8.000000e+00f)) * Scales[((((((((((k_0 * 32) + ((((int)threadIdx.x) & 3) * 8)) + (ax0_0 * 4)) + ax0_1) >> 7) * 4096) + (((int)blockIdx.x) * 128)) + (ax0_ax1_0_fused_0_0_2 * 32)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))]);
        }
      }
      *(uint4*)(B_decode_shared + (((((ax0_ax1_0_fused_0_0_2 * 1024) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(B_decode_local + 0);
    }
    __syncthreads();
    call_cutlass_mma_prologue(C_cutlass_warp_mma, (&(A_shared[((k_0 & 1) * 512)])), (&(B_decode_shared[0])), 32, 32);
    call_cutlass_mma_body(C_cutlass_warp_mma);
    call_cutlass_mma_epilogue(C_cutlass_warp_mma);
  }
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  for (int ax0_ax1_0_fused_0_0_3 = 0; ax0_ax1_0_fused_0_0_3 < 4; ++ax0_ax1_0_fused_0_0_3) {
    for (int ax0_0_1 = 0; ax0_0_1 < 2; ++ax0_0_1) {
      *(char2*)(B_local_1 + 0) = *(char2*)(B_shared + (((((ax0_ax1_0_fused_0_0_3 * 512) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + (ax0_0_1 * 2)) + 2048));
      for (int ax0_1_1 = 0; ax0_1_1 < 4; ++ax0_1_1) {
        B_decode_local_1[((ax0_0_1 * 4) + ax0_1_1)] = ((((half)((B_local_1[(ax0_1_1 >> 1)] >> ((signed char)((ax0_1_1 & 1) * 4))) & (signed char)15)) - __float2half_rn(8.000000e+00f)) * Scales[(((((((((((((int)threadIdx.x) & 3) * 8) + (ax0_0_1 * 4)) + ax0_1_1) + 10976) >> 7) * 4096) + (((int)blockIdx.x) * 128)) + (ax0_ax1_0_fused_0_0_3 * 32)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))]);
      }
    }
    *(uint4*)(B_decode_shared + (((((ax0_ax1_0_fused_0_0_3 * 1024) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(B_decode_local_1 + 0);
  }
  __syncthreads();
  call_cutlass_mma_prologue(C_cutlass_warp_mma, (&(A_shared[512])), (&(B_decode_shared[0])), 32, 32);
  call_cutlass_mma_body(C_cutlass_warp_mma);
  call_cutlass_mma_epilogue(C_cutlass_warp_mma);
  #pragma unroll
  for (int ax1_0 = 0; ax1_0 < 8; ++ax1_0) {
    *(uint1*)(C + (((((((ax1_0 & 1) * 32768) + ((((int)threadIdx.x) >> 2) * 4096)) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.y) * 32)) + ((ax1_0 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(C_cutlass_warp_mma + (ax1_0 * 2));
  }
}



 __global__ void __launch_bounds__(32) cutlass_kernel_fp16_nf4_fp16_m1n12288k4096_nt_1x8(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ C) {
            int8_t* B = ((int8_t *)QB);
	 half* Scales = (half *)((int8_t *)QB + 25165824);                 
            // const dim3 GridDim(12288, 1, 1);
            // const dim3 BlockDim(32, 1, 1);
            // cutlass_kernel_fp16_nf4_fp16_m1n12288k4096_nt_1x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
  half in_thread_C_local[1];
  half A_local[8];
  signed char B_local[4];
  half red_buf0[1];
  in_thread_C_local[0] = __float2half_rn(0.000000e+00f);
  for (int k_0 = 0; k_0 < 16; ++k_0) {
    *(uint4*)(A_local + 0) = *(uint4*)(A + ((k_0 * 256) + (((int)threadIdx.x) * 8)));
    *(int*)(B_local + 0) = *(int*)(B + (((((int)blockIdx.x) * 2048) + (k_0 * 128)) + (((int)threadIdx.x) * 4)));
    for (int k_2 = 0; k_2 < 8; ++k_2) {
      in_thread_C_local[0] = (in_thread_C_local[0] + (A_local[k_2] * ((((half)((B_local[(k_2 >> 1)] >> ((signed char)((k_2 & 1) * 4))) & (signed char)15)) - __float2half_rn(8.000000e+00f)) * Scales[(((k_0 * 24576) + ((((int)threadIdx.x) >> 4) * 12288)) + ((int)blockIdx.x))])));
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



 __global__ void __launch_bounds__(32) cutlass_kernel_fp16_nf4_fp16_m1n4096k4096_nt_1x8(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ C) {
            int8_t* B = ((int8_t *)QB);
	 half* Scales = (half *)((int8_t *)QB + 8388608);                 
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
      in_thread_C_local[0] = (in_thread_C_local[0] + (A_local[k_2] * ((((half)((B_local[(k_2 >> 1)] >> ((signed char)((k_2 & 1) * 4))) & (signed char)15)) - __float2half_rn(8.000000e+00f)) * Scales[(((k_0 * 8192) + ((((int)threadIdx.x) >> 4) * 4096)) + ((int)blockIdx.x))])));
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
	 half* Scales = (half *)((int8_t *)QB + 22544384);                 
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
      in_thread_C_local[0] = (in_thread_C_local[0] + (A_local[k_2] * ((((half)((B_local[(k_2 >> 1)] >> ((signed char)((k_2 & 1) * 4))) & (signed char)15)) - __float2half_rn(8.000000e+00f)) * Scales[(((k_0 * 22016) + ((((int)threadIdx.x) >> 4) * 11008)) + ((int)blockIdx.x))])));
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
	 half* Scales = (half *)((int8_t *)QB + 22544384);                 
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
      in_thread_C_local[0] = (in_thread_C_local[0] + (A_local[k_2] * ((((half)((B_local[(k_2 >> 1)] >> ((signed char)((k_2 & 1) * 4))) & (signed char)15)) - __float2half_rn(8.000000e+00f)) * Scales[(((k_0 * 8192) + ((((int)threadIdx.x) >> 4) * 4096)) + ((int)blockIdx.x))])));
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
    
        if (M == 16 && N == 12288 && K == 4096){
            
             const dim3 GridDim(256, 1, 1);
             const dim3 BlockDim(32, 3, 1);
             cutlass_kernel_fp16_nf4_fp16_m16n12288k4096_nt_16x48x32_16x8x16_<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
        if (M == 16 && N == 4096 && K == 4096){
            
             const dim3 GridDim(256, 1, 1);
             const dim3 BlockDim(32, 2, 1);
             cutlass_kernel_fp16_nf4_fp16_m16n4096k4096_nt_16x16x32_16x8x16_<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
        if (M == 16 && N == 11008 && K == 4096){
            
             const dim3 GridDim(172, 1, 1);
             const dim3 BlockDim(32, 4, 1);
             cutlass_kernel_fp16_nf4_fp16_m16n11008k4096_nt_16x64x32_16x8x16_<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
        if (M == 16 && N == 4096 && K == 11008){
            
             const dim3 GridDim(32, 1, 1);
             const dim3 BlockDim(32, 4, 1);
             cutlass_kernel_fp16_nf4_fp16_m16n4096k11008_nt_16x128x32_16x8x16_<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
        if (M == 1 && N == 12288 && K == 4096){
            
             const dim3 GridDim(12288, 1, 1);
             const dim3 BlockDim(32, 1, 1);
             cutlass_kernel_fp16_nf4_fp16_m1n12288k4096_nt_1x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
        if (M == 1 && N == 4096 && K == 4096){
            
             const dim3 GridDim(4096, 1, 1);
             const dim3 BlockDim(32, 1, 1);
             cutlass_kernel_fp16_nf4_fp16_m1n4096k4096_nt_1x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
        if (M == 1 && N == 11008 && K == 4096){
            
             const dim3 GridDim(11008, 1, 1);
             const dim3 BlockDim(32, 1, 1);
             cutlass_kernel_fp16_nf4_fp16_m1n11008k4096_nt_1x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
        if (M == 1 && N == 4096 && K == 11008){
            
             const dim3 GridDim(4096, 1, 1);
             const dim3 BlockDim(32, 1, 1);
             cutlass_kernel_fp16_nf4_fp16_m1n4096k11008_nt_1x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
    return -1;
}
