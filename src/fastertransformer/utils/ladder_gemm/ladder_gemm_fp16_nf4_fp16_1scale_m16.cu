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

__global__ void __launch_bounds__(128) cutlass_kernel_fp16_nf4_fp16_m16n12288k4096_nt_16x32x32_16x8x16_(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ C) {
            int8_t* B = ((int8_t *)QB);
	 half* LUT = (half *)((int8_t *)QB + 25165824);
	 half* Scales = (half *)((int8_t *)QB + 25165824 + 32);                 
            // const dim3 GridDim(384, 1, 1);
            // const dim3 BlockDim(32, 4, 1);
            // cutlass_kernel_fp16_nf4_fp16_m16n12288k4096_nt_16x32x32_16x8x16_<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
  
  __shared__ half LUT_shared[16];
    __shared__ half A_shared[1024];
  __shared__ half B_decode_shared[2560];
  __shared__ signed char B_shared[1280];
  half B_decode_local[8];
  signed char B_local[2];
  half B_decode_local_1[8];
  signed char B_local_1[2];
  if (((int)threadIdx.x) < 16) {
    LUT_shared[((int)threadIdx.x)] = LUT[((int)threadIdx.x)];
  }
  ALLOCATE_CUTLASS_OBJECT(C_cutlass_warp_mma, cutlass::gemm::warp::GemmTensorOp<
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<16, 32>,
    cutlass::layout::ColumnMajor
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
  #pragma unroll
  for (int ax0_ax1_0_fused_0_0 = 0; ax0_ax1_0_fused_0_0 < 1; ++ax0_ax1_0_fused_0_0) {
    __syncthreads();

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
        :: "r"(addr), "l"((void*)(B + ((((((int)blockIdx.x) * 65536) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 2) * 2048)) + ((((int)threadIdx.x) & 3) * 4)))), "n"(4)
    );
  }
    __syncthreads();
    for (int ax0_0 = 0; ax0_0 < 2; ++ax0_0) {
      *(char2*)(B_local + 0) = *(char2*)(B_shared + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 4)) + (ax0_0 * 2)));
      uint2 __1;
        int4 __2;
        int __3;
          int __4;
            int4 __5;
              int4 __6 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __7 = make_int4(2, 2, 2, 2);
              __5.x = (__6.x%__7.x);
              __5.y = (__6.y%__7.y);
              __5.z = (__6.z%__7.z);
              __5.w = (__6.w%__7.w);
            int4 __8;
              int4 __9 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __10 = make_int4(2, 2, 2, 2);
              __8.x = (__9.x/__10.x);
              __8.y = (__9.y/__10.y);
              __8.z = (__9.z/__10.z);
              __8.w = (__9.w/__10.w);
            int4 __11;
            ushort4 __12;
              ushort4 __13;
                ushort4 __14;
                  int4 __15 = make_int4(2, 2, 2, 2);
                  int4 __16 = make_int4(0, 0, 0, 0);
                  __14.x = (__15.x>=__16.x);
                  __14.y = (__15.y>=__16.y);
                  __14.z = (__15.z>=__16.z);
                  __14.w = (__15.w>=__16.w);
                ushort4 __17;
                  int4 __18 = make_int4(0, 0, 0, 0);
                  __17.x = (__5.x>=__18.x);
                  __17.y = (__5.y>=__18.y);
                  __17.z = (__5.z>=__18.z);
                  __17.w = (__5.w>=__18.w);
                __13.x = (__14.x&&__17.x);
                __13.y = (__14.y&&__17.y);
                __13.z = (__14.z&&__17.z);
                __13.w = (__14.w&&__17.w);
              ushort4 __19;
                ushort4 __20;
                  int4 __21 = make_int4(2, 2, 2, 2);
                  int4 __22 = make_int4(0, 0, 0, 0);
                  __20.x = (__21.x<__22.x);
                  __20.y = (__21.y<__22.y);
                  __20.z = (__21.z<__22.z);
                  __20.w = (__21.w<__22.w);
                ushort4 __23;
                  int4 __24 = make_int4(0, 0, 0, 0);
                  __23.x = (__5.x<=__24.x);
                  __23.y = (__5.y<=__24.y);
                  __23.z = (__5.z<=__24.z);
                  __23.w = (__5.w<=__24.w);
                __19.x = (__20.x&&__23.x);
                __19.y = (__20.y&&__23.y);
                __19.z = (__20.z&&__23.z);
                __19.w = (__20.w&&__23.w);
              __12.x = (__13.x||__19.x);
              __12.y = (__13.y||__19.y);
              __12.z = (__13.z||__19.z);
              __12.w = (__13.w||__19.w);
            int4 __25;
              int4 __26 = make_int4(1, 1, 1, 1);
              __25.x = (__8.x-__26.x);
              __25.y = (__8.y-__26.y);
              __25.z = (__8.z-__26.z);
              __25.w = (__8.w-__26.w);
            __11.x = (bool(__12.x)?__8.x:__25.x);
            __11.y = (bool(__12.y)?__8.y:__25.y);
            __11.z = (bool(__12.z)?__8.z:__25.z);
            __11.w = (bool(__12.w)?__8.w:__25.w);
            int __27 = ((0x000000ff << 0) & (B_local[__11.x] << 0))|((0x000000ff << 8) & (B_local[__11.y] << 8))|((0x000000ff << 16) & (B_local[__11.z] << 16))|((0x000000ff << 24) & (B_local[__11.w] << 24));
            int __28;
            int4 __29;
              int4 __30;
                int4 __31 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
                int4 __32 = make_int4(2, 2, 2, 2);
                __30.x = (__31.x%__32.x);
                __30.y = (__31.y%__32.y);
                __30.z = (__31.z%__32.z);
                __30.w = (__31.w%__32.w);
              int4 __33;
              ushort4 __34;
                ushort4 __35;
                  ushort4 __36;
                    int4 __37 = make_int4(2, 2, 2, 2);
                    int4 __38 = make_int4(0, 0, 0, 0);
                    __36.x = (__37.x>=__38.x);
                    __36.y = (__37.y>=__38.y);
                    __36.z = (__37.z>=__38.z);
                    __36.w = (__37.w>=__38.w);
                  ushort4 __39;
                    int4 __40 = make_int4(0, 0, 0, 0);
                    __39.x = (__30.x>=__40.x);
                    __39.y = (__30.y>=__40.y);
                    __39.z = (__30.z>=__40.z);
                    __39.w = (__30.w>=__40.w);
                  __35.x = (__36.x&&__39.x);
                  __35.y = (__36.y&&__39.y);
                  __35.z = (__36.z&&__39.z);
                  __35.w = (__36.w&&__39.w);
                ushort4 __41;
                  ushort4 __42;
                    int4 __43 = make_int4(2, 2, 2, 2);
                    int4 __44 = make_int4(0, 0, 0, 0);
                    __42.x = (__43.x<__44.x);
                    __42.y = (__43.y<__44.y);
                    __42.z = (__43.z<__44.z);
                    __42.w = (__43.w<__44.w);
                  ushort4 __45;
                    int4 __46 = make_int4(0, 0, 0, 0);
                    __45.x = (__30.x<=__46.x);
                    __45.y = (__30.y<=__46.y);
                    __45.z = (__30.z<=__46.z);
                    __45.w = (__30.w<=__46.w);
                  __41.x = (__42.x&&__45.x);
                  __41.y = (__42.y&&__45.y);
                  __41.z = (__42.z&&__45.z);
                  __41.w = (__42.w&&__45.w);
                __34.x = (__35.x||__41.x);
                __34.y = (__35.y||__41.y);
                __34.z = (__35.z||__41.z);
                __34.w = (__35.w||__41.w);
              int4 __47;
                int4 __48 = make_int4(2, 2, 2, 2);
                __47.x = (__30.x+__48.x);
                __47.y = (__30.y+__48.y);
                __47.z = (__30.z+__48.z);
                __47.w = (__30.w+__48.w);
              __33.x = (bool(__34.x)?__30.x:__47.x);
              __33.y = (bool(__34.y)?__30.y:__47.y);
              __33.z = (bool(__34.z)?__30.z:__47.z);
              __33.w = (bool(__34.w)?__30.w:__47.w);
              int4 __49 = make_int4(4, 4, 4, 4);
              __29.x = (__33.x*__49.x);
              __29.y = (__33.y*__49.y);
              __29.z = (__33.z*__49.z);
              __29.w = (__33.w*__49.w);
            __28=((signed char)(__29.x) << 0);
            __28=__28 & ~(0x000000ff << 8) |((signed char)(__29.y) << 8);
            __28=__28 & ~(0x000000ff << 16) |((signed char)(__29.z) << 16);
            __28=__28 & ~(0x000000ff << 24) |((signed char)(__29.w) << 24);
            __4=((((char)(__27 >> 0)) >> ((char)(__28 >> 0))) << 0);
            __4=__4 & ~(0x000000ff << 8) |((((char)(__27 >> 8)) >> ((char)(__28 >> 8))) << 8);
            __4=__4 & ~(0x000000ff << 16) |((((char)(__27 >> 16)) >> ((char)(__28 >> 16))) << 16);
            __4=__4 & ~(0x000000ff << 24) |((((char)(__27 >> 24)) >> ((char)(__28 >> 24))) << 24);
          int __50 = (int)252645135;
          __3=((((char)(__4 >> 0)) & ((char)(__50 >> 0))) << 0);
          __3=__3 & ~(0x000000ff << 8) |((((char)(__4 >> 8)) & ((char)(__50 >> 8))) << 8);
          __3=__3 & ~(0x000000ff << 16) |((((char)(__4 >> 16)) & ((char)(__50 >> 16))) << 16);
          __3=__3 & ~(0x000000ff << 24) |((((char)(__4 >> 24)) & ((char)(__50 >> 24))) << 24);
        __2.x = (int)(((char)(__3 >> 0)));
        __2.y = (int)(((char)(__3 >> 8)));
        __2.z = (int)(((char)(__3 >> 16)));
        __2.w = (int)(((char)(__3 >> 24)));
        uint2 __51 = make_uint2(__pack_half2(LUT_shared[__2.x],LUT_shared[__2.y]),__pack_half2(LUT_shared[__2.z],LUT_shared[__2.w]));
        uint2 __52 = make_uint2(__pack_half2(Scales[(((((int)blockIdx.x) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))], Scales[(((((int)blockIdx.x) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))]), __pack_half2(Scales[(((((int)blockIdx.x) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))], Scales[(((((int)blockIdx.x) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))]));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(__51.x)))->x*((half2*)(&(__52.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(__51.x)))->y*((half2*)(&(__52.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(__51.y)))->x*((half2*)(&(__52.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(__51.y)))->y*((half2*)(&(__52.y)))->y);
      *(uint2*)(B_decode_local + (ax0_0 * 4)) = __1;
    }
    *(uint4*)(B_decode_shared + (((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) = *(uint4*)(B_decode_local + 0);
  }
__asm__ __volatile__("cp.async.commit_group;");

  #pragma unroll
  for (int ax0_ax1_fused_0_0_0_1 = 0; ax0_ax1_fused_0_0_0_1 < 1; ++ax0_ax1_fused_0_0_0_1) {
    if (((int)threadIdx.y) < 2) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + (((((((int)threadIdx.y) * 256) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 512))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(A_shared + (((((((int)threadIdx.y) * 256) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 512)))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((int)threadIdx.y) * 32768) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 32))), "n"(16)
    );
  }
    }
  }
  #pragma unroll
  for (int ax0_ax1_0_fused_0_0_1 = 0; ax0_ax1_0_fused_0_0_1 < 1; ++ax0_ax1_0_fused_0_0_1) {
    __syncthreads();

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
        :: "r"(addr), "l"((void*)(B + (((((((int)blockIdx.x) * 65536) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 2) * 2048)) + ((((int)threadIdx.x) & 3) * 4)) + 16))), "n"(4)
    );
  }
    __syncthreads();
    for (int ax0_0_1 = 0; ax0_0_1 < 2; ++ax0_0_1) {
      *(char2*)(B_local + 0) = *(char2*)(B_shared + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 4)) + (ax0_0_1 * 2)));
      uint2 __53;
        int4 __54;
        int __55;
          int __56;
            int4 __57;
              int4 __58 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __59 = make_int4(2, 2, 2, 2);
              __57.x = (__58.x%__59.x);
              __57.y = (__58.y%__59.y);
              __57.z = (__58.z%__59.z);
              __57.w = (__58.w%__59.w);
            int4 __60;
              int4 __61 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __62 = make_int4(2, 2, 2, 2);
              __60.x = (__61.x/__62.x);
              __60.y = (__61.y/__62.y);
              __60.z = (__61.z/__62.z);
              __60.w = (__61.w/__62.w);
            int4 __63;
            ushort4 __64;
              ushort4 __65;
                ushort4 __66;
                  int4 __67 = make_int4(2, 2, 2, 2);
                  int4 __68 = make_int4(0, 0, 0, 0);
                  __66.x = (__67.x>=__68.x);
                  __66.y = (__67.y>=__68.y);
                  __66.z = (__67.z>=__68.z);
                  __66.w = (__67.w>=__68.w);
                ushort4 __69;
                  int4 __70 = make_int4(0, 0, 0, 0);
                  __69.x = (__57.x>=__70.x);
                  __69.y = (__57.y>=__70.y);
                  __69.z = (__57.z>=__70.z);
                  __69.w = (__57.w>=__70.w);
                __65.x = (__66.x&&__69.x);
                __65.y = (__66.y&&__69.y);
                __65.z = (__66.z&&__69.z);
                __65.w = (__66.w&&__69.w);
              ushort4 __71;
                ushort4 __72;
                  int4 __73 = make_int4(2, 2, 2, 2);
                  int4 __74 = make_int4(0, 0, 0, 0);
                  __72.x = (__73.x<__74.x);
                  __72.y = (__73.y<__74.y);
                  __72.z = (__73.z<__74.z);
                  __72.w = (__73.w<__74.w);
                ushort4 __75;
                  int4 __76 = make_int4(0, 0, 0, 0);
                  __75.x = (__57.x<=__76.x);
                  __75.y = (__57.y<=__76.y);
                  __75.z = (__57.z<=__76.z);
                  __75.w = (__57.w<=__76.w);
                __71.x = (__72.x&&__75.x);
                __71.y = (__72.y&&__75.y);
                __71.z = (__72.z&&__75.z);
                __71.w = (__72.w&&__75.w);
              __64.x = (__65.x||__71.x);
              __64.y = (__65.y||__71.y);
              __64.z = (__65.z||__71.z);
              __64.w = (__65.w||__71.w);
            int4 __77;
              int4 __78 = make_int4(1, 1, 1, 1);
              __77.x = (__60.x-__78.x);
              __77.y = (__60.y-__78.y);
              __77.z = (__60.z-__78.z);
              __77.w = (__60.w-__78.w);
            __63.x = (bool(__64.x)?__60.x:__77.x);
            __63.y = (bool(__64.y)?__60.y:__77.y);
            __63.z = (bool(__64.z)?__60.z:__77.z);
            __63.w = (bool(__64.w)?__60.w:__77.w);
            int __79 = ((0x000000ff << 0) & (B_local[__63.x] << 0))|((0x000000ff << 8) & (B_local[__63.y] << 8))|((0x000000ff << 16) & (B_local[__63.z] << 16))|((0x000000ff << 24) & (B_local[__63.w] << 24));
            int __80;
            int4 __81;
              int4 __82;
                int4 __83 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
                int4 __84 = make_int4(2, 2, 2, 2);
                __82.x = (__83.x%__84.x);
                __82.y = (__83.y%__84.y);
                __82.z = (__83.z%__84.z);
                __82.w = (__83.w%__84.w);
              int4 __85;
              ushort4 __86;
                ushort4 __87;
                  ushort4 __88;
                    int4 __89 = make_int4(2, 2, 2, 2);
                    int4 __90 = make_int4(0, 0, 0, 0);
                    __88.x = (__89.x>=__90.x);
                    __88.y = (__89.y>=__90.y);
                    __88.z = (__89.z>=__90.z);
                    __88.w = (__89.w>=__90.w);
                  ushort4 __91;
                    int4 __92 = make_int4(0, 0, 0, 0);
                    __91.x = (__82.x>=__92.x);
                    __91.y = (__82.y>=__92.y);
                    __91.z = (__82.z>=__92.z);
                    __91.w = (__82.w>=__92.w);
                  __87.x = (__88.x&&__91.x);
                  __87.y = (__88.y&&__91.y);
                  __87.z = (__88.z&&__91.z);
                  __87.w = (__88.w&&__91.w);
                ushort4 __93;
                  ushort4 __94;
                    int4 __95 = make_int4(2, 2, 2, 2);
                    int4 __96 = make_int4(0, 0, 0, 0);
                    __94.x = (__95.x<__96.x);
                    __94.y = (__95.y<__96.y);
                    __94.z = (__95.z<__96.z);
                    __94.w = (__95.w<__96.w);
                  ushort4 __97;
                    int4 __98 = make_int4(0, 0, 0, 0);
                    __97.x = (__82.x<=__98.x);
                    __97.y = (__82.y<=__98.y);
                    __97.z = (__82.z<=__98.z);
                    __97.w = (__82.w<=__98.w);
                  __93.x = (__94.x&&__97.x);
                  __93.y = (__94.y&&__97.y);
                  __93.z = (__94.z&&__97.z);
                  __93.w = (__94.w&&__97.w);
                __86.x = (__87.x||__93.x);
                __86.y = (__87.y||__93.y);
                __86.z = (__87.z||__93.z);
                __86.w = (__87.w||__93.w);
              int4 __99;
                int4 __100 = make_int4(2, 2, 2, 2);
                __99.x = (__82.x+__100.x);
                __99.y = (__82.y+__100.y);
                __99.z = (__82.z+__100.z);
                __99.w = (__82.w+__100.w);
              __85.x = (bool(__86.x)?__82.x:__99.x);
              __85.y = (bool(__86.y)?__82.y:__99.y);
              __85.z = (bool(__86.z)?__82.z:__99.z);
              __85.w = (bool(__86.w)?__82.w:__99.w);
              int4 __101 = make_int4(4, 4, 4, 4);
              __81.x = (__85.x*__101.x);
              __81.y = (__85.y*__101.y);
              __81.z = (__85.z*__101.z);
              __81.w = (__85.w*__101.w);
            __80=((signed char)(__81.x) << 0);
            __80=__80 & ~(0x000000ff << 8) |((signed char)(__81.y) << 8);
            __80=__80 & ~(0x000000ff << 16) |((signed char)(__81.z) << 16);
            __80=__80 & ~(0x000000ff << 24) |((signed char)(__81.w) << 24);
            __56=((((char)(__79 >> 0)) >> ((char)(__80 >> 0))) << 0);
            __56=__56 & ~(0x000000ff << 8) |((((char)(__79 >> 8)) >> ((char)(__80 >> 8))) << 8);
            __56=__56 & ~(0x000000ff << 16) |((((char)(__79 >> 16)) >> ((char)(__80 >> 16))) << 16);
            __56=__56 & ~(0x000000ff << 24) |((((char)(__79 >> 24)) >> ((char)(__80 >> 24))) << 24);
          int __102 = (int)252645135;
          __55=((((char)(__56 >> 0)) & ((char)(__102 >> 0))) << 0);
          __55=__55 & ~(0x000000ff << 8) |((((char)(__56 >> 8)) & ((char)(__102 >> 8))) << 8);
          __55=__55 & ~(0x000000ff << 16) |((((char)(__56 >> 16)) & ((char)(__102 >> 16))) << 16);
          __55=__55 & ~(0x000000ff << 24) |((((char)(__56 >> 24)) & ((char)(__102 >> 24))) << 24);
        __54.x = (int)(((char)(__55 >> 0)));
        __54.y = (int)(((char)(__55 >> 8)));
        __54.z = (int)(((char)(__55 >> 16)));
        __54.w = (int)(((char)(__55 >> 24)));
        uint2 __103 = make_uint2(__pack_half2(LUT_shared[__54.x],LUT_shared[__54.y]),__pack_half2(LUT_shared[__54.z],LUT_shared[__54.w]));
        uint2 __104 = make_uint2(__pack_half2(Scales[(((((int)blockIdx.x) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))], Scales[(((((int)blockIdx.x) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))]), __pack_half2(Scales[(((((int)blockIdx.x) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))], Scales[(((((int)blockIdx.x) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))]));
        ((half2*)(&(__53.x)))->x = (((half2*)(&(__103.x)))->x*((half2*)(&(__104.x)))->x);
        ((half2*)(&(__53.x)))->y = (((half2*)(&(__103.x)))->y*((half2*)(&(__104.x)))->y);
        ((half2*)(&(__53.y)))->x = (((half2*)(&(__103.y)))->x*((half2*)(&(__104.y)))->x);
        ((half2*)(&(__53.y)))->y = (((half2*)(&(__103.y)))->y*((half2*)(&(__104.y)))->y);
      *(uint2*)(B_decode_local + (ax0_0_1 * 4)) = __53;
    }
    *(uint4*)(B_decode_shared + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 1280)) = *(uint4*)(B_decode_local + 0);
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 1;");

  __syncthreads();
  call_cutlass_mma_prologue(C_cutlass_warp_mma, (&(A_shared[0])), (&(B_decode_shared[0])), 32, 40);
  call_cutlass_mma_body(C_cutlass_warp_mma);
  for (int k_0 = 0; k_0 < 126; ++k_0) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_0_0_0_2 = 0; ax0_ax1_fused_0_0_0_2 < 1; ++ax0_ax1_fused_0_0_0_2) {
      if (((int)threadIdx.y) < 2) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + ((((((k_0 & 1) * 512) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(A_shared + ((((((k_0 & 1) * 512) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + (((((((int)threadIdx.y) * 32768) + ((((int)threadIdx.x) >> 2) * 4096)) + (k_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 64))), "n"(16)
    );
  }
      }
    }
    #pragma unroll
    for (int ax0_ax1_0_fused_0_0_2 = 0; ax0_ax1_0_fused_0_0_2 < 1; ++ax0_ax1_0_fused_0_0_2) {
      __syncthreads();

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
        :: "r"(addr), "l"((void*)(B + ((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 2) * 2048)) + (k_0 * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 32))), "n"(4)
    );
  }
      __syncthreads();
      for (int ax0_0_2 = 0; ax0_0_2 < 2; ++ax0_0_2) {
        *(char2*)(B_local_1 + 0) = *(char2*)(B_shared + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 4)) + (ax0_0_2 * 2)));
        uint2 __105;
          int4 __106;
          int __107;
            int __108;
              int4 __109;
                int4 __110 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
                int4 __111 = make_int4(2, 2, 2, 2);
                __109.x = (__110.x%__111.x);
                __109.y = (__110.y%__111.y);
                __109.z = (__110.z%__111.z);
                __109.w = (__110.w%__111.w);
              int4 __112;
                int4 __113 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
                int4 __114 = make_int4(2, 2, 2, 2);
                __112.x = (__113.x/__114.x);
                __112.y = (__113.y/__114.y);
                __112.z = (__113.z/__114.z);
                __112.w = (__113.w/__114.w);
              int4 __115;
              ushort4 __116;
                ushort4 __117;
                  ushort4 __118;
                    int4 __119 = make_int4(2, 2, 2, 2);
                    int4 __120 = make_int4(0, 0, 0, 0);
                    __118.x = (__119.x>=__120.x);
                    __118.y = (__119.y>=__120.y);
                    __118.z = (__119.z>=__120.z);
                    __118.w = (__119.w>=__120.w);
                  ushort4 __121;
                    int4 __122 = make_int4(0, 0, 0, 0);
                    __121.x = (__109.x>=__122.x);
                    __121.y = (__109.y>=__122.y);
                    __121.z = (__109.z>=__122.z);
                    __121.w = (__109.w>=__122.w);
                  __117.x = (__118.x&&__121.x);
                  __117.y = (__118.y&&__121.y);
                  __117.z = (__118.z&&__121.z);
                  __117.w = (__118.w&&__121.w);
                ushort4 __123;
                  ushort4 __124;
                    int4 __125 = make_int4(2, 2, 2, 2);
                    int4 __126 = make_int4(0, 0, 0, 0);
                    __124.x = (__125.x<__126.x);
                    __124.y = (__125.y<__126.y);
                    __124.z = (__125.z<__126.z);
                    __124.w = (__125.w<__126.w);
                  ushort4 __127;
                    int4 __128 = make_int4(0, 0, 0, 0);
                    __127.x = (__109.x<=__128.x);
                    __127.y = (__109.y<=__128.y);
                    __127.z = (__109.z<=__128.z);
                    __127.w = (__109.w<=__128.w);
                  __123.x = (__124.x&&__127.x);
                  __123.y = (__124.y&&__127.y);
                  __123.z = (__124.z&&__127.z);
                  __123.w = (__124.w&&__127.w);
                __116.x = (__117.x||__123.x);
                __116.y = (__117.y||__123.y);
                __116.z = (__117.z||__123.z);
                __116.w = (__117.w||__123.w);
              int4 __129;
                int4 __130 = make_int4(1, 1, 1, 1);
                __129.x = (__112.x-__130.x);
                __129.y = (__112.y-__130.y);
                __129.z = (__112.z-__130.z);
                __129.w = (__112.w-__130.w);
              __115.x = (bool(__116.x)?__112.x:__129.x);
              __115.y = (bool(__116.y)?__112.y:__129.y);
              __115.z = (bool(__116.z)?__112.z:__129.z);
              __115.w = (bool(__116.w)?__112.w:__129.w);
              int __131 = ((0x000000ff << 0) & (B_local_1[__115.x] << 0))|((0x000000ff << 8) & (B_local_1[__115.y] << 8))|((0x000000ff << 16) & (B_local_1[__115.z] << 16))|((0x000000ff << 24) & (B_local_1[__115.w] << 24));
              int __132;
              int4 __133;
                int4 __134;
                  int4 __135 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
                  int4 __136 = make_int4(2, 2, 2, 2);
                  __134.x = (__135.x%__136.x);
                  __134.y = (__135.y%__136.y);
                  __134.z = (__135.z%__136.z);
                  __134.w = (__135.w%__136.w);
                int4 __137;
                ushort4 __138;
                  ushort4 __139;
                    ushort4 __140;
                      int4 __141 = make_int4(2, 2, 2, 2);
                      int4 __142 = make_int4(0, 0, 0, 0);
                      __140.x = (__141.x>=__142.x);
                      __140.y = (__141.y>=__142.y);
                      __140.z = (__141.z>=__142.z);
                      __140.w = (__141.w>=__142.w);
                    ushort4 __143;
                      int4 __144 = make_int4(0, 0, 0, 0);
                      __143.x = (__134.x>=__144.x);
                      __143.y = (__134.y>=__144.y);
                      __143.z = (__134.z>=__144.z);
                      __143.w = (__134.w>=__144.w);
                    __139.x = (__140.x&&__143.x);
                    __139.y = (__140.y&&__143.y);
                    __139.z = (__140.z&&__143.z);
                    __139.w = (__140.w&&__143.w);
                  ushort4 __145;
                    ushort4 __146;
                      int4 __147 = make_int4(2, 2, 2, 2);
                      int4 __148 = make_int4(0, 0, 0, 0);
                      __146.x = (__147.x<__148.x);
                      __146.y = (__147.y<__148.y);
                      __146.z = (__147.z<__148.z);
                      __146.w = (__147.w<__148.w);
                    ushort4 __149;
                      int4 __150 = make_int4(0, 0, 0, 0);
                      __149.x = (__134.x<=__150.x);
                      __149.y = (__134.y<=__150.y);
                      __149.z = (__134.z<=__150.z);
                      __149.w = (__134.w<=__150.w);
                    __145.x = (__146.x&&__149.x);
                    __145.y = (__146.y&&__149.y);
                    __145.z = (__146.z&&__149.z);
                    __145.w = (__146.w&&__149.w);
                  __138.x = (__139.x||__145.x);
                  __138.y = (__139.y||__145.y);
                  __138.z = (__139.z||__145.z);
                  __138.w = (__139.w||__145.w);
                int4 __151;
                  int4 __152 = make_int4(2, 2, 2, 2);
                  __151.x = (__134.x+__152.x);
                  __151.y = (__134.y+__152.y);
                  __151.z = (__134.z+__152.z);
                  __151.w = (__134.w+__152.w);
                __137.x = (bool(__138.x)?__134.x:__151.x);
                __137.y = (bool(__138.y)?__134.y:__151.y);
                __137.z = (bool(__138.z)?__134.z:__151.z);
                __137.w = (bool(__138.w)?__134.w:__151.w);
                int4 __153 = make_int4(4, 4, 4, 4);
                __133.x = (__137.x*__153.x);
                __133.y = (__137.y*__153.y);
                __133.z = (__137.z*__153.z);
                __133.w = (__137.w*__153.w);
              __132=((signed char)(__133.x) << 0);
              __132=__132 & ~(0x000000ff << 8) |((signed char)(__133.y) << 8);
              __132=__132 & ~(0x000000ff << 16) |((signed char)(__133.z) << 16);
              __132=__132 & ~(0x000000ff << 24) |((signed char)(__133.w) << 24);
              __108=((((char)(__131 >> 0)) >> ((char)(__132 >> 0))) << 0);
              __108=__108 & ~(0x000000ff << 8) |((((char)(__131 >> 8)) >> ((char)(__132 >> 8))) << 8);
              __108=__108 & ~(0x000000ff << 16) |((((char)(__131 >> 16)) >> ((char)(__132 >> 16))) << 16);
              __108=__108 & ~(0x000000ff << 24) |((((char)(__131 >> 24)) >> ((char)(__132 >> 24))) << 24);
            int __154 = (int)252645135;
            __107=((((char)(__108 >> 0)) & ((char)(__154 >> 0))) << 0);
            __107=__107 & ~(0x000000ff << 8) |((((char)(__108 >> 8)) & ((char)(__154 >> 8))) << 8);
            __107=__107 & ~(0x000000ff << 16) |((((char)(__108 >> 16)) & ((char)(__154 >> 16))) << 16);
            __107=__107 & ~(0x000000ff << 24) |((((char)(__108 >> 24)) & ((char)(__154 >> 24))) << 24);
          __106.x = (int)(((char)(__107 >> 0)));
          __106.y = (int)(((char)(__107 >> 8)));
          __106.z = (int)(((char)(__107 >> 16)));
          __106.w = (int)(((char)(__107 >> 24)));
          uint2 __155 = make_uint2(__pack_half2(LUT_shared[__106.x],LUT_shared[__106.y]),__pack_half2(LUT_shared[__106.z],LUT_shared[__106.w]));
          int4 __156 = make_int4((((((((((k_0 * 32) + ((((int)threadIdx.x) & 3) * 8)) + (ax0_0_2 * 4)) + 64) >> 7) * 12288) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)), (((((((((k_0 * 32) + ((((int)threadIdx.x) & 3) * 8)) + (ax0_0_2 * 4)) + 64) >> 7) * 12288) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)), (((((((((k_0 * 32) + ((((int)threadIdx.x) & 3) * 8)) + (ax0_0_2 * 4)) + 64) >> 7) * 12288) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)), (((((((((k_0 * 32) + ((((int)threadIdx.x) & 3) * 8)) + (ax0_0_2 * 4)) + 64) >> 7) * 12288) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)));
          uint2 __157 = make_uint2(__pack_half2(Scales[__156.x],Scales[__156.y]),__pack_half2(Scales[__156.z],Scales[__156.w]));
          ((half2*)(&(__105.x)))->x = (((half2*)(&(__155.x)))->x*((half2*)(&(__157.x)))->x);
          ((half2*)(&(__105.x)))->y = (((half2*)(&(__155.x)))->y*((half2*)(&(__157.x)))->y);
          ((half2*)(&(__105.y)))->x = (((half2*)(&(__155.y)))->x*((half2*)(&(__157.y)))->x);
          ((half2*)(&(__105.y)))->y = (((half2*)(&(__155.y)))->y*((half2*)(&(__157.y)))->y);
        *(uint2*)(B_decode_local_1 + (ax0_0_2 * 4)) = __105;
      }
      *(uint4*)(B_decode_shared + (((((k_0 & 1) * 1280) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) = *(uint4*)(B_decode_local_1 + 0);
    }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 1;");

    __syncthreads();
    call_cutlass_mma_prologue(C_cutlass_warp_mma, (&(A_shared[(((k_0 + 1) & 1) * 512)])), (&(B_decode_shared[(((k_0 + 1) & 1) * 1280)])), 32, 40);
    call_cutlass_mma_epilogue(C_cutlass_warp_mma);
    call_cutlass_mma_body(C_cutlass_warp_mma);
  }
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  call_cutlass_mma_prologue(C_cutlass_warp_mma, (&(A_shared[512])), (&(B_decode_shared[1280])), 32, 40);
  call_cutlass_mma_epilogue(C_cutlass_warp_mma);
  call_cutlass_mma_body(C_cutlass_warp_mma);
  call_cutlass_mma_epilogue(C_cutlass_warp_mma);
  #pragma unroll
  for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0) {
    *(uint1*)(C + (((((ax1_0 * 98304) + ((((int)threadIdx.x) >> 2) * 12288)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.y) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(C_cutlass_warp_mma + (ax1_0 * 2));
  }
}



__global__ void __launch_bounds__(32) cutlass_kernel_fp16_nf4_fp16_m16n4096k4096_nt_8x32x32_8x32x16_(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ C) {
            int8_t* B = ((int8_t *)QB);
	 half* LUT = (half *)((int8_t *)QB + 8388608);
	 half* Scales = (half *)((int8_t *)QB + 8388608 + 32);                 
            // const dim3 GridDim(256, 1, 1);
            // const dim3 BlockDim(32, 1, 1);
            // cutlass_kernel_fp16_nf4_fp16_m16n4096k4096_nt_8x32x32_8x32x16_<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
  
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, half> C_wmma_accumulator[1];
  __shared__ half A_shared[320];
  __shared__ half B_decode_shared[1280];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, half, nvcuda::wmma::row_major> A_shared_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, half, nvcuda::wmma::col_major> B_decode_shared_wmma_matrix_b[1];
  nvcuda::wmma::fill_fragment(C_wmma_accumulator[0], __float2half_rn(0.000000e+00f));
  for (int k_outer = 0; k_outer < 128; ++k_outer) {
    __syncthreads();
    *(uint4*)(A_shared + (((((int)threadIdx.x) >> 2) * 40) + ((((int)threadIdx.x) & 3) * 8))) = *(uint4*)(A + (((((((int)blockIdx.x) >> 7) * 32768) + ((((int)threadIdx.x) >> 2) * 4096)) + (k_outer * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    uint2 __1;
      int4 __2;
      int __3;
        int __4;
          int4 __5;
            int4 __6 = make_int4((((((((int)blockIdx.x) & 127) * 65536) + ((((int)threadIdx.x) >> 3) * 2048)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)), (((((((int)blockIdx.x) & 127) * 65536) + ((((int)threadIdx.x) >> 3) * 2048)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)), (((((((int)blockIdx.x) & 127) * 65536) + ((((int)threadIdx.x) >> 3) * 2048)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)), (((((((int)blockIdx.x) & 127) * 65536) + ((((int)threadIdx.x) >> 3) * 2048)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)));
            int4 __7;
              int4 __8 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __9 = make_int4(2, 2, 2, 2);
              __7.x = (__8.x%__9.x);
              __7.y = (__8.y%__9.y);
              __7.z = (__8.z%__9.z);
              __7.w = (__8.w%__9.w);
            int4 __10;
              int4 __11 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __12 = make_int4(2, 2, 2, 2);
              __10.x = (__11.x/__12.x);
              __10.y = (__11.y/__12.y);
              __10.z = (__11.z/__12.z);
              __10.w = (__11.w/__12.w);
            int4 __13;
            ushort4 __14;
              ushort4 __15;
                ushort4 __16;
                  int4 __17 = make_int4(2, 2, 2, 2);
                  int4 __18 = make_int4(0, 0, 0, 0);
                  __16.x = (__17.x>=__18.x);
                  __16.y = (__17.y>=__18.y);
                  __16.z = (__17.z>=__18.z);
                  __16.w = (__17.w>=__18.w);
                ushort4 __19;
                  int4 __20 = make_int4(0, 0, 0, 0);
                  __19.x = (__7.x>=__20.x);
                  __19.y = (__7.y>=__20.y);
                  __19.z = (__7.z>=__20.z);
                  __19.w = (__7.w>=__20.w);
                __15.x = (__16.x&&__19.x);
                __15.y = (__16.y&&__19.y);
                __15.z = (__16.z&&__19.z);
                __15.w = (__16.w&&__19.w);
              ushort4 __21;
                ushort4 __22;
                  int4 __23 = make_int4(2, 2, 2, 2);
                  int4 __24 = make_int4(0, 0, 0, 0);
                  __22.x = (__23.x<__24.x);
                  __22.y = (__23.y<__24.y);
                  __22.z = (__23.z<__24.z);
                  __22.w = (__23.w<__24.w);
                ushort4 __25;
                  int4 __26 = make_int4(0, 0, 0, 0);
                  __25.x = (__7.x<=__26.x);
                  __25.y = (__7.y<=__26.y);
                  __25.z = (__7.z<=__26.z);
                  __25.w = (__7.w<=__26.w);
                __21.x = (__22.x&&__25.x);
                __21.y = (__22.y&&__25.y);
                __21.z = (__22.z&&__25.z);
                __21.w = (__22.w&&__25.w);
              __14.x = (__15.x||__21.x);
              __14.y = (__15.y||__21.y);
              __14.z = (__15.z||__21.z);
              __14.w = (__15.w||__21.w);
            int4 __27;
              int4 __28 = make_int4(1, 1, 1, 1);
              __27.x = (__10.x-__28.x);
              __27.y = (__10.y-__28.y);
              __27.z = (__10.z-__28.z);
              __27.w = (__10.w-__28.w);
            __13.x = (bool(__14.x)?__10.x:__27.x);
            __13.y = (bool(__14.y)?__10.y:__27.y);
            __13.z = (bool(__14.z)?__10.z:__27.z);
            __13.w = (bool(__14.w)?__10.w:__27.w);
            __5.x = (__6.x+__13.x);
            __5.y = (__6.y+__13.y);
            __5.z = (__6.z+__13.z);
            __5.w = (__6.w+__13.w);
          int __29 = ((0x000000ff << 0) & (B[__5.x] << 0))|((0x000000ff << 8) & (B[__5.y] << 8))|((0x000000ff << 16) & (B[__5.z] << 16))|((0x000000ff << 24) & (B[__5.w] << 24));
          int __30;
          int4 __31;
            int4 __32;
              int4 __33 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __34 = make_int4(2, 2, 2, 2);
              __32.x = (__33.x%__34.x);
              __32.y = (__33.y%__34.y);
              __32.z = (__33.z%__34.z);
              __32.w = (__33.w%__34.w);
            int4 __35;
            ushort4 __36;
              ushort4 __37;
                ushort4 __38;
                  int4 __39 = make_int4(2, 2, 2, 2);
                  int4 __40 = make_int4(0, 0, 0, 0);
                  __38.x = (__39.x>=__40.x);
                  __38.y = (__39.y>=__40.y);
                  __38.z = (__39.z>=__40.z);
                  __38.w = (__39.w>=__40.w);
                ushort4 __41;
                  int4 __42 = make_int4(0, 0, 0, 0);
                  __41.x = (__32.x>=__42.x);
                  __41.y = (__32.y>=__42.y);
                  __41.z = (__32.z>=__42.z);
                  __41.w = (__32.w>=__42.w);
                __37.x = (__38.x&&__41.x);
                __37.y = (__38.y&&__41.y);
                __37.z = (__38.z&&__41.z);
                __37.w = (__38.w&&__41.w);
              ushort4 __43;
                ushort4 __44;
                  int4 __45 = make_int4(2, 2, 2, 2);
                  int4 __46 = make_int4(0, 0, 0, 0);
                  __44.x = (__45.x<__46.x);
                  __44.y = (__45.y<__46.y);
                  __44.z = (__45.z<__46.z);
                  __44.w = (__45.w<__46.w);
                ushort4 __47;
                  int4 __48 = make_int4(0, 0, 0, 0);
                  __47.x = (__32.x<=__48.x);
                  __47.y = (__32.y<=__48.y);
                  __47.z = (__32.z<=__48.z);
                  __47.w = (__32.w<=__48.w);
                __43.x = (__44.x&&__47.x);
                __43.y = (__44.y&&__47.y);
                __43.z = (__44.z&&__47.z);
                __43.w = (__44.w&&__47.w);
              __36.x = (__37.x||__43.x);
              __36.y = (__37.y||__43.y);
              __36.z = (__37.z||__43.z);
              __36.w = (__37.w||__43.w);
            int4 __49;
              int4 __50 = make_int4(2, 2, 2, 2);
              __49.x = (__32.x+__50.x);
              __49.y = (__32.y+__50.y);
              __49.z = (__32.z+__50.z);
              __49.w = (__32.w+__50.w);
            __35.x = (bool(__36.x)?__32.x:__49.x);
            __35.y = (bool(__36.y)?__32.y:__49.y);
            __35.z = (bool(__36.z)?__32.z:__49.z);
            __35.w = (bool(__36.w)?__32.w:__49.w);
            int4 __51 = make_int4(4, 4, 4, 4);
            __31.x = (__35.x*__51.x);
            __31.y = (__35.y*__51.y);
            __31.z = (__35.z*__51.z);
            __31.w = (__35.w*__51.w);
          __30=((signed char)(__31.x) << 0);
          __30=__30 & ~(0x000000ff << 8) |((signed char)(__31.y) << 8);
          __30=__30 & ~(0x000000ff << 16) |((signed char)(__31.z) << 16);
          __30=__30 & ~(0x000000ff << 24) |((signed char)(__31.w) << 24);
          __4=((((char)(__29 >> 0)) >> ((char)(__30 >> 0))) << 0);
          __4=__4 & ~(0x000000ff << 8) |((((char)(__29 >> 8)) >> ((char)(__30 >> 8))) << 8);
          __4=__4 & ~(0x000000ff << 16) |((((char)(__29 >> 16)) >> ((char)(__30 >> 16))) << 16);
          __4=__4 & ~(0x000000ff << 24) |((((char)(__29 >> 24)) >> ((char)(__30 >> 24))) << 24);
        int __52 = (int)252645135;
        __3=((((char)(__4 >> 0)) & ((char)(__52 >> 0))) << 0);
        __3=__3 & ~(0x000000ff << 8) |((((char)(__4 >> 8)) & ((char)(__52 >> 8))) << 8);
        __3=__3 & ~(0x000000ff << 16) |((((char)(__4 >> 16)) & ((char)(__52 >> 16))) << 16);
        __3=__3 & ~(0x000000ff << 24) |((((char)(__4 >> 24)) & ((char)(__52 >> 24))) << 24);
      __2.x = (int)(((char)(__3 >> 0)));
      __2.y = (int)(((char)(__3 >> 8)));
      __2.z = (int)(((char)(__3 >> 16)));
      __2.w = (int)(((char)(__3 >> 24)));
      uint2 __53 = make_uint2(__pack_half2(LUT[__2.x],LUT[__2.y]),__pack_half2(LUT[__2.z],LUT[__2.w]));
      uint2 __54 = make_uint2(__pack_half2(Scales[((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3))], Scales[((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3))]), __pack_half2(Scales[((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3))], Scales[((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3))]));
      ((half2*)(&(__1.x)))->x = (((half2*)(&(__53.x)))->x*((half2*)(&(__54.x)))->x);
      ((half2*)(&(__1.x)))->y = (((half2*)(&(__53.x)))->y*((half2*)(&(__54.x)))->y);
      ((half2*)(&(__1.y)))->x = (((half2*)(&(__53.y)))->x*((half2*)(&(__54.y)))->x);
      ((half2*)(&(__1.y)))->y = (((half2*)(&(__53.y)))->y*((half2*)(&(__54.y)))->y);
    *(uint2*)(B_decode_shared + (((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 4))) = __1;
    uint2 __55;
      int4 __56;
      int __57;
        int __58;
          int4 __59;
            int4 __60 = make_int4(((((((((int)blockIdx.x) & 127) * 65536) + ((((int)threadIdx.x) >> 3) * 2048)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 8192), ((((((((int)blockIdx.x) & 127) * 65536) + ((((int)threadIdx.x) >> 3) * 2048)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 8192), ((((((((int)blockIdx.x) & 127) * 65536) + ((((int)threadIdx.x) >> 3) * 2048)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 8192), ((((((((int)blockIdx.x) & 127) * 65536) + ((((int)threadIdx.x) >> 3) * 2048)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 8192));
            int4 __61;
              int4 __62 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __63 = make_int4(2, 2, 2, 2);
              __61.x = (__62.x%__63.x);
              __61.y = (__62.y%__63.y);
              __61.z = (__62.z%__63.z);
              __61.w = (__62.w%__63.w);
            int4 __64;
              int4 __65 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __66 = make_int4(2, 2, 2, 2);
              __64.x = (__65.x/__66.x);
              __64.y = (__65.y/__66.y);
              __64.z = (__65.z/__66.z);
              __64.w = (__65.w/__66.w);
            int4 __67;
            ushort4 __68;
              ushort4 __69;
                ushort4 __70;
                  int4 __71 = make_int4(2, 2, 2, 2);
                  int4 __72 = make_int4(0, 0, 0, 0);
                  __70.x = (__71.x>=__72.x);
                  __70.y = (__71.y>=__72.y);
                  __70.z = (__71.z>=__72.z);
                  __70.w = (__71.w>=__72.w);
                ushort4 __73;
                  int4 __74 = make_int4(0, 0, 0, 0);
                  __73.x = (__61.x>=__74.x);
                  __73.y = (__61.y>=__74.y);
                  __73.z = (__61.z>=__74.z);
                  __73.w = (__61.w>=__74.w);
                __69.x = (__70.x&&__73.x);
                __69.y = (__70.y&&__73.y);
                __69.z = (__70.z&&__73.z);
                __69.w = (__70.w&&__73.w);
              ushort4 __75;
                ushort4 __76;
                  int4 __77 = make_int4(2, 2, 2, 2);
                  int4 __78 = make_int4(0, 0, 0, 0);
                  __76.x = (__77.x<__78.x);
                  __76.y = (__77.y<__78.y);
                  __76.z = (__77.z<__78.z);
                  __76.w = (__77.w<__78.w);
                ushort4 __79;
                  int4 __80 = make_int4(0, 0, 0, 0);
                  __79.x = (__61.x<=__80.x);
                  __79.y = (__61.y<=__80.y);
                  __79.z = (__61.z<=__80.z);
                  __79.w = (__61.w<=__80.w);
                __75.x = (__76.x&&__79.x);
                __75.y = (__76.y&&__79.y);
                __75.z = (__76.z&&__79.z);
                __75.w = (__76.w&&__79.w);
              __68.x = (__69.x||__75.x);
              __68.y = (__69.y||__75.y);
              __68.z = (__69.z||__75.z);
              __68.w = (__69.w||__75.w);
            int4 __81;
              int4 __82 = make_int4(1, 1, 1, 1);
              __81.x = (__64.x-__82.x);
              __81.y = (__64.y-__82.y);
              __81.z = (__64.z-__82.z);
              __81.w = (__64.w-__82.w);
            __67.x = (bool(__68.x)?__64.x:__81.x);
            __67.y = (bool(__68.y)?__64.y:__81.y);
            __67.z = (bool(__68.z)?__64.z:__81.z);
            __67.w = (bool(__68.w)?__64.w:__81.w);
            __59.x = (__60.x+__67.x);
            __59.y = (__60.y+__67.y);
            __59.z = (__60.z+__67.z);
            __59.w = (__60.w+__67.w);
          int __83 = ((0x000000ff << 0) & (B[__59.x] << 0))|((0x000000ff << 8) & (B[__59.y] << 8))|((0x000000ff << 16) & (B[__59.z] << 16))|((0x000000ff << 24) & (B[__59.w] << 24));
          int __84;
          int4 __85;
            int4 __86;
              int4 __87 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __88 = make_int4(2, 2, 2, 2);
              __86.x = (__87.x%__88.x);
              __86.y = (__87.y%__88.y);
              __86.z = (__87.z%__88.z);
              __86.w = (__87.w%__88.w);
            int4 __89;
            ushort4 __90;
              ushort4 __91;
                ushort4 __92;
                  int4 __93 = make_int4(2, 2, 2, 2);
                  int4 __94 = make_int4(0, 0, 0, 0);
                  __92.x = (__93.x>=__94.x);
                  __92.y = (__93.y>=__94.y);
                  __92.z = (__93.z>=__94.z);
                  __92.w = (__93.w>=__94.w);
                ushort4 __95;
                  int4 __96 = make_int4(0, 0, 0, 0);
                  __95.x = (__86.x>=__96.x);
                  __95.y = (__86.y>=__96.y);
                  __95.z = (__86.z>=__96.z);
                  __95.w = (__86.w>=__96.w);
                __91.x = (__92.x&&__95.x);
                __91.y = (__92.y&&__95.y);
                __91.z = (__92.z&&__95.z);
                __91.w = (__92.w&&__95.w);
              ushort4 __97;
                ushort4 __98;
                  int4 __99 = make_int4(2, 2, 2, 2);
                  int4 __100 = make_int4(0, 0, 0, 0);
                  __98.x = (__99.x<__100.x);
                  __98.y = (__99.y<__100.y);
                  __98.z = (__99.z<__100.z);
                  __98.w = (__99.w<__100.w);
                ushort4 __101;
                  int4 __102 = make_int4(0, 0, 0, 0);
                  __101.x = (__86.x<=__102.x);
                  __101.y = (__86.y<=__102.y);
                  __101.z = (__86.z<=__102.z);
                  __101.w = (__86.w<=__102.w);
                __97.x = (__98.x&&__101.x);
                __97.y = (__98.y&&__101.y);
                __97.z = (__98.z&&__101.z);
                __97.w = (__98.w&&__101.w);
              __90.x = (__91.x||__97.x);
              __90.y = (__91.y||__97.y);
              __90.z = (__91.z||__97.z);
              __90.w = (__91.w||__97.w);
            int4 __103;
              int4 __104 = make_int4(2, 2, 2, 2);
              __103.x = (__86.x+__104.x);
              __103.y = (__86.y+__104.y);
              __103.z = (__86.z+__104.z);
              __103.w = (__86.w+__104.w);
            __89.x = (bool(__90.x)?__86.x:__103.x);
            __89.y = (bool(__90.y)?__86.y:__103.y);
            __89.z = (bool(__90.z)?__86.z:__103.z);
            __89.w = (bool(__90.w)?__86.w:__103.w);
            int4 __105 = make_int4(4, 4, 4, 4);
            __85.x = (__89.x*__105.x);
            __85.y = (__89.y*__105.y);
            __85.z = (__89.z*__105.z);
            __85.w = (__89.w*__105.w);
          __84=((signed char)(__85.x) << 0);
          __84=__84 & ~(0x000000ff << 8) |((signed char)(__85.y) << 8);
          __84=__84 & ~(0x000000ff << 16) |((signed char)(__85.z) << 16);
          __84=__84 & ~(0x000000ff << 24) |((signed char)(__85.w) << 24);
          __58=((((char)(__83 >> 0)) >> ((char)(__84 >> 0))) << 0);
          __58=__58 & ~(0x000000ff << 8) |((((char)(__83 >> 8)) >> ((char)(__84 >> 8))) << 8);
          __58=__58 & ~(0x000000ff << 16) |((((char)(__83 >> 16)) >> ((char)(__84 >> 16))) << 16);
          __58=__58 & ~(0x000000ff << 24) |((((char)(__83 >> 24)) >> ((char)(__84 >> 24))) << 24);
        int __106 = (int)252645135;
        __57=((((char)(__58 >> 0)) & ((char)(__106 >> 0))) << 0);
        __57=__57 & ~(0x000000ff << 8) |((((char)(__58 >> 8)) & ((char)(__106 >> 8))) << 8);
        __57=__57 & ~(0x000000ff << 16) |((((char)(__58 >> 16)) & ((char)(__106 >> 16))) << 16);
        __57=__57 & ~(0x000000ff << 24) |((((char)(__58 >> 24)) & ((char)(__106 >> 24))) << 24);
      __56.x = (int)(((char)(__57 >> 0)));
      __56.y = (int)(((char)(__57 >> 8)));
      __56.z = (int)(((char)(__57 >> 16)));
      __56.w = (int)(((char)(__57 >> 24)));
      uint2 __107 = make_uint2(__pack_half2(LUT[__56.x],LUT[__56.y]),__pack_half2(LUT[__56.z],LUT[__56.w]));
      uint2 __108 = make_uint2(__pack_half2(Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 4)], Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 4)]), __pack_half2(Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 4)], Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 4)]));
      ((half2*)(&(__55.x)))->x = (((half2*)(&(__107.x)))->x*((half2*)(&(__108.x)))->x);
      ((half2*)(&(__55.x)))->y = (((half2*)(&(__107.x)))->y*((half2*)(&(__108.x)))->y);
      ((half2*)(&(__55.y)))->x = (((half2*)(&(__107.y)))->x*((half2*)(&(__108.y)))->x);
      ((half2*)(&(__55.y)))->y = (((half2*)(&(__107.y)))->y*((half2*)(&(__108.y)))->y);
    *(uint2*)(B_decode_shared + ((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 4)) + 160)) = __55;
    uint2 __109;
      int4 __110;
      int __111;
        int __112;
          int4 __113;
            int4 __114 = make_int4(((((((((int)blockIdx.x) & 127) * 65536) + ((((int)threadIdx.x) >> 3) * 2048)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 16384), ((((((((int)blockIdx.x) & 127) * 65536) + ((((int)threadIdx.x) >> 3) * 2048)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 16384), ((((((((int)blockIdx.x) & 127) * 65536) + ((((int)threadIdx.x) >> 3) * 2048)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 16384), ((((((((int)blockIdx.x) & 127) * 65536) + ((((int)threadIdx.x) >> 3) * 2048)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 16384));
            int4 __115;
              int4 __116 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __117 = make_int4(2, 2, 2, 2);
              __115.x = (__116.x%__117.x);
              __115.y = (__116.y%__117.y);
              __115.z = (__116.z%__117.z);
              __115.w = (__116.w%__117.w);
            int4 __118;
              int4 __119 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __120 = make_int4(2, 2, 2, 2);
              __118.x = (__119.x/__120.x);
              __118.y = (__119.y/__120.y);
              __118.z = (__119.z/__120.z);
              __118.w = (__119.w/__120.w);
            int4 __121;
            ushort4 __122;
              ushort4 __123;
                ushort4 __124;
                  int4 __125 = make_int4(2, 2, 2, 2);
                  int4 __126 = make_int4(0, 0, 0, 0);
                  __124.x = (__125.x>=__126.x);
                  __124.y = (__125.y>=__126.y);
                  __124.z = (__125.z>=__126.z);
                  __124.w = (__125.w>=__126.w);
                ushort4 __127;
                  int4 __128 = make_int4(0, 0, 0, 0);
                  __127.x = (__115.x>=__128.x);
                  __127.y = (__115.y>=__128.y);
                  __127.z = (__115.z>=__128.z);
                  __127.w = (__115.w>=__128.w);
                __123.x = (__124.x&&__127.x);
                __123.y = (__124.y&&__127.y);
                __123.z = (__124.z&&__127.z);
                __123.w = (__124.w&&__127.w);
              ushort4 __129;
                ushort4 __130;
                  int4 __131 = make_int4(2, 2, 2, 2);
                  int4 __132 = make_int4(0, 0, 0, 0);
                  __130.x = (__131.x<__132.x);
                  __130.y = (__131.y<__132.y);
                  __130.z = (__131.z<__132.z);
                  __130.w = (__131.w<__132.w);
                ushort4 __133;
                  int4 __134 = make_int4(0, 0, 0, 0);
                  __133.x = (__115.x<=__134.x);
                  __133.y = (__115.y<=__134.y);
                  __133.z = (__115.z<=__134.z);
                  __133.w = (__115.w<=__134.w);
                __129.x = (__130.x&&__133.x);
                __129.y = (__130.y&&__133.y);
                __129.z = (__130.z&&__133.z);
                __129.w = (__130.w&&__133.w);
              __122.x = (__123.x||__129.x);
              __122.y = (__123.y||__129.y);
              __122.z = (__123.z||__129.z);
              __122.w = (__123.w||__129.w);
            int4 __135;
              int4 __136 = make_int4(1, 1, 1, 1);
              __135.x = (__118.x-__136.x);
              __135.y = (__118.y-__136.y);
              __135.z = (__118.z-__136.z);
              __135.w = (__118.w-__136.w);
            __121.x = (bool(__122.x)?__118.x:__135.x);
            __121.y = (bool(__122.y)?__118.y:__135.y);
            __121.z = (bool(__122.z)?__118.z:__135.z);
            __121.w = (bool(__122.w)?__118.w:__135.w);
            __113.x = (__114.x+__121.x);
            __113.y = (__114.y+__121.y);
            __113.z = (__114.z+__121.z);
            __113.w = (__114.w+__121.w);
          int __137 = ((0x000000ff << 0) & (B[__113.x] << 0))|((0x000000ff << 8) & (B[__113.y] << 8))|((0x000000ff << 16) & (B[__113.z] << 16))|((0x000000ff << 24) & (B[__113.w] << 24));
          int __138;
          int4 __139;
            int4 __140;
              int4 __141 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __142 = make_int4(2, 2, 2, 2);
              __140.x = (__141.x%__142.x);
              __140.y = (__141.y%__142.y);
              __140.z = (__141.z%__142.z);
              __140.w = (__141.w%__142.w);
            int4 __143;
            ushort4 __144;
              ushort4 __145;
                ushort4 __146;
                  int4 __147 = make_int4(2, 2, 2, 2);
                  int4 __148 = make_int4(0, 0, 0, 0);
                  __146.x = (__147.x>=__148.x);
                  __146.y = (__147.y>=__148.y);
                  __146.z = (__147.z>=__148.z);
                  __146.w = (__147.w>=__148.w);
                ushort4 __149;
                  int4 __150 = make_int4(0, 0, 0, 0);
                  __149.x = (__140.x>=__150.x);
                  __149.y = (__140.y>=__150.y);
                  __149.z = (__140.z>=__150.z);
                  __149.w = (__140.w>=__150.w);
                __145.x = (__146.x&&__149.x);
                __145.y = (__146.y&&__149.y);
                __145.z = (__146.z&&__149.z);
                __145.w = (__146.w&&__149.w);
              ushort4 __151;
                ushort4 __152;
                  int4 __153 = make_int4(2, 2, 2, 2);
                  int4 __154 = make_int4(0, 0, 0, 0);
                  __152.x = (__153.x<__154.x);
                  __152.y = (__153.y<__154.y);
                  __152.z = (__153.z<__154.z);
                  __152.w = (__153.w<__154.w);
                ushort4 __155;
                  int4 __156 = make_int4(0, 0, 0, 0);
                  __155.x = (__140.x<=__156.x);
                  __155.y = (__140.y<=__156.y);
                  __155.z = (__140.z<=__156.z);
                  __155.w = (__140.w<=__156.w);
                __151.x = (__152.x&&__155.x);
                __151.y = (__152.y&&__155.y);
                __151.z = (__152.z&&__155.z);
                __151.w = (__152.w&&__155.w);
              __144.x = (__145.x||__151.x);
              __144.y = (__145.y||__151.y);
              __144.z = (__145.z||__151.z);
              __144.w = (__145.w||__151.w);
            int4 __157;
              int4 __158 = make_int4(2, 2, 2, 2);
              __157.x = (__140.x+__158.x);
              __157.y = (__140.y+__158.y);
              __157.z = (__140.z+__158.z);
              __157.w = (__140.w+__158.w);
            __143.x = (bool(__144.x)?__140.x:__157.x);
            __143.y = (bool(__144.y)?__140.y:__157.y);
            __143.z = (bool(__144.z)?__140.z:__157.z);
            __143.w = (bool(__144.w)?__140.w:__157.w);
            int4 __159 = make_int4(4, 4, 4, 4);
            __139.x = (__143.x*__159.x);
            __139.y = (__143.y*__159.y);
            __139.z = (__143.z*__159.z);
            __139.w = (__143.w*__159.w);
          __138=((signed char)(__139.x) << 0);
          __138=__138 & ~(0x000000ff << 8) |((signed char)(__139.y) << 8);
          __138=__138 & ~(0x000000ff << 16) |((signed char)(__139.z) << 16);
          __138=__138 & ~(0x000000ff << 24) |((signed char)(__139.w) << 24);
          __112=((((char)(__137 >> 0)) >> ((char)(__138 >> 0))) << 0);
          __112=__112 & ~(0x000000ff << 8) |((((char)(__137 >> 8)) >> ((char)(__138 >> 8))) << 8);
          __112=__112 & ~(0x000000ff << 16) |((((char)(__137 >> 16)) >> ((char)(__138 >> 16))) << 16);
          __112=__112 & ~(0x000000ff << 24) |((((char)(__137 >> 24)) >> ((char)(__138 >> 24))) << 24);
        int __160 = (int)252645135;
        __111=((((char)(__112 >> 0)) & ((char)(__160 >> 0))) << 0);
        __111=__111 & ~(0x000000ff << 8) |((((char)(__112 >> 8)) & ((char)(__160 >> 8))) << 8);
        __111=__111 & ~(0x000000ff << 16) |((((char)(__112 >> 16)) & ((char)(__160 >> 16))) << 16);
        __111=__111 & ~(0x000000ff << 24) |((((char)(__112 >> 24)) & ((char)(__160 >> 24))) << 24);
      __110.x = (int)(((char)(__111 >> 0)));
      __110.y = (int)(((char)(__111 >> 8)));
      __110.z = (int)(((char)(__111 >> 16)));
      __110.w = (int)(((char)(__111 >> 24)));
      uint2 __161 = make_uint2(__pack_half2(LUT[__110.x],LUT[__110.y]),__pack_half2(LUT[__110.z],LUT[__110.w]));
      uint2 __162 = make_uint2(__pack_half2(Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 8)], Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 8)]), __pack_half2(Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 8)], Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 8)]));
      ((half2*)(&(__109.x)))->x = (((half2*)(&(__161.x)))->x*((half2*)(&(__162.x)))->x);
      ((half2*)(&(__109.x)))->y = (((half2*)(&(__161.x)))->y*((half2*)(&(__162.x)))->y);
      ((half2*)(&(__109.y)))->x = (((half2*)(&(__161.y)))->x*((half2*)(&(__162.y)))->x);
      ((half2*)(&(__109.y)))->y = (((half2*)(&(__161.y)))->y*((half2*)(&(__162.y)))->y);
    *(uint2*)(B_decode_shared + ((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 4)) + 320)) = __109;
    uint2 __163;
      int4 __164;
      int __165;
        int __166;
          int4 __167;
            int4 __168 = make_int4(((((((((int)blockIdx.x) & 127) * 65536) + ((((int)threadIdx.x) >> 3) * 2048)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 24576), ((((((((int)blockIdx.x) & 127) * 65536) + ((((int)threadIdx.x) >> 3) * 2048)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 24576), ((((((((int)blockIdx.x) & 127) * 65536) + ((((int)threadIdx.x) >> 3) * 2048)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 24576), ((((((((int)blockIdx.x) & 127) * 65536) + ((((int)threadIdx.x) >> 3) * 2048)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 24576));
            int4 __169;
              int4 __170 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __171 = make_int4(2, 2, 2, 2);
              __169.x = (__170.x%__171.x);
              __169.y = (__170.y%__171.y);
              __169.z = (__170.z%__171.z);
              __169.w = (__170.w%__171.w);
            int4 __172;
              int4 __173 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __174 = make_int4(2, 2, 2, 2);
              __172.x = (__173.x/__174.x);
              __172.y = (__173.y/__174.y);
              __172.z = (__173.z/__174.z);
              __172.w = (__173.w/__174.w);
            int4 __175;
            ushort4 __176;
              ushort4 __177;
                ushort4 __178;
                  int4 __179 = make_int4(2, 2, 2, 2);
                  int4 __180 = make_int4(0, 0, 0, 0);
                  __178.x = (__179.x>=__180.x);
                  __178.y = (__179.y>=__180.y);
                  __178.z = (__179.z>=__180.z);
                  __178.w = (__179.w>=__180.w);
                ushort4 __181;
                  int4 __182 = make_int4(0, 0, 0, 0);
                  __181.x = (__169.x>=__182.x);
                  __181.y = (__169.y>=__182.y);
                  __181.z = (__169.z>=__182.z);
                  __181.w = (__169.w>=__182.w);
                __177.x = (__178.x&&__181.x);
                __177.y = (__178.y&&__181.y);
                __177.z = (__178.z&&__181.z);
                __177.w = (__178.w&&__181.w);
              ushort4 __183;
                ushort4 __184;
                  int4 __185 = make_int4(2, 2, 2, 2);
                  int4 __186 = make_int4(0, 0, 0, 0);
                  __184.x = (__185.x<__186.x);
                  __184.y = (__185.y<__186.y);
                  __184.z = (__185.z<__186.z);
                  __184.w = (__185.w<__186.w);
                ushort4 __187;
                  int4 __188 = make_int4(0, 0, 0, 0);
                  __187.x = (__169.x<=__188.x);
                  __187.y = (__169.y<=__188.y);
                  __187.z = (__169.z<=__188.z);
                  __187.w = (__169.w<=__188.w);
                __183.x = (__184.x&&__187.x);
                __183.y = (__184.y&&__187.y);
                __183.z = (__184.z&&__187.z);
                __183.w = (__184.w&&__187.w);
              __176.x = (__177.x||__183.x);
              __176.y = (__177.y||__183.y);
              __176.z = (__177.z||__183.z);
              __176.w = (__177.w||__183.w);
            int4 __189;
              int4 __190 = make_int4(1, 1, 1, 1);
              __189.x = (__172.x-__190.x);
              __189.y = (__172.y-__190.y);
              __189.z = (__172.z-__190.z);
              __189.w = (__172.w-__190.w);
            __175.x = (bool(__176.x)?__172.x:__189.x);
            __175.y = (bool(__176.y)?__172.y:__189.y);
            __175.z = (bool(__176.z)?__172.z:__189.z);
            __175.w = (bool(__176.w)?__172.w:__189.w);
            __167.x = (__168.x+__175.x);
            __167.y = (__168.y+__175.y);
            __167.z = (__168.z+__175.z);
            __167.w = (__168.w+__175.w);
          int __191 = ((0x000000ff << 0) & (B[__167.x] << 0))|((0x000000ff << 8) & (B[__167.y] << 8))|((0x000000ff << 16) & (B[__167.z] << 16))|((0x000000ff << 24) & (B[__167.w] << 24));
          int __192;
          int4 __193;
            int4 __194;
              int4 __195 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __196 = make_int4(2, 2, 2, 2);
              __194.x = (__195.x%__196.x);
              __194.y = (__195.y%__196.y);
              __194.z = (__195.z%__196.z);
              __194.w = (__195.w%__196.w);
            int4 __197;
            ushort4 __198;
              ushort4 __199;
                ushort4 __200;
                  int4 __201 = make_int4(2, 2, 2, 2);
                  int4 __202 = make_int4(0, 0, 0, 0);
                  __200.x = (__201.x>=__202.x);
                  __200.y = (__201.y>=__202.y);
                  __200.z = (__201.z>=__202.z);
                  __200.w = (__201.w>=__202.w);
                ushort4 __203;
                  int4 __204 = make_int4(0, 0, 0, 0);
                  __203.x = (__194.x>=__204.x);
                  __203.y = (__194.y>=__204.y);
                  __203.z = (__194.z>=__204.z);
                  __203.w = (__194.w>=__204.w);
                __199.x = (__200.x&&__203.x);
                __199.y = (__200.y&&__203.y);
                __199.z = (__200.z&&__203.z);
                __199.w = (__200.w&&__203.w);
              ushort4 __205;
                ushort4 __206;
                  int4 __207 = make_int4(2, 2, 2, 2);
                  int4 __208 = make_int4(0, 0, 0, 0);
                  __206.x = (__207.x<__208.x);
                  __206.y = (__207.y<__208.y);
                  __206.z = (__207.z<__208.z);
                  __206.w = (__207.w<__208.w);
                ushort4 __209;
                  int4 __210 = make_int4(0, 0, 0, 0);
                  __209.x = (__194.x<=__210.x);
                  __209.y = (__194.y<=__210.y);
                  __209.z = (__194.z<=__210.z);
                  __209.w = (__194.w<=__210.w);
                __205.x = (__206.x&&__209.x);
                __205.y = (__206.y&&__209.y);
                __205.z = (__206.z&&__209.z);
                __205.w = (__206.w&&__209.w);
              __198.x = (__199.x||__205.x);
              __198.y = (__199.y||__205.y);
              __198.z = (__199.z||__205.z);
              __198.w = (__199.w||__205.w);
            int4 __211;
              int4 __212 = make_int4(2, 2, 2, 2);
              __211.x = (__194.x+__212.x);
              __211.y = (__194.y+__212.y);
              __211.z = (__194.z+__212.z);
              __211.w = (__194.w+__212.w);
            __197.x = (bool(__198.x)?__194.x:__211.x);
            __197.y = (bool(__198.y)?__194.y:__211.y);
            __197.z = (bool(__198.z)?__194.z:__211.z);
            __197.w = (bool(__198.w)?__194.w:__211.w);
            int4 __213 = make_int4(4, 4, 4, 4);
            __193.x = (__197.x*__213.x);
            __193.y = (__197.y*__213.y);
            __193.z = (__197.z*__213.z);
            __193.w = (__197.w*__213.w);
          __192=((signed char)(__193.x) << 0);
          __192=__192 & ~(0x000000ff << 8) |((signed char)(__193.y) << 8);
          __192=__192 & ~(0x000000ff << 16) |((signed char)(__193.z) << 16);
          __192=__192 & ~(0x000000ff << 24) |((signed char)(__193.w) << 24);
          __166=((((char)(__191 >> 0)) >> ((char)(__192 >> 0))) << 0);
          __166=__166 & ~(0x000000ff << 8) |((((char)(__191 >> 8)) >> ((char)(__192 >> 8))) << 8);
          __166=__166 & ~(0x000000ff << 16) |((((char)(__191 >> 16)) >> ((char)(__192 >> 16))) << 16);
          __166=__166 & ~(0x000000ff << 24) |((((char)(__191 >> 24)) >> ((char)(__192 >> 24))) << 24);
        int __214 = (int)252645135;
        __165=((((char)(__166 >> 0)) & ((char)(__214 >> 0))) << 0);
        __165=__165 & ~(0x000000ff << 8) |((((char)(__166 >> 8)) & ((char)(__214 >> 8))) << 8);
        __165=__165 & ~(0x000000ff << 16) |((((char)(__166 >> 16)) & ((char)(__214 >> 16))) << 16);
        __165=__165 & ~(0x000000ff << 24) |((((char)(__166 >> 24)) & ((char)(__214 >> 24))) << 24);
      __164.x = (int)(((char)(__165 >> 0)));
      __164.y = (int)(((char)(__165 >> 8)));
      __164.z = (int)(((char)(__165 >> 16)));
      __164.w = (int)(((char)(__165 >> 24)));
      uint2 __215 = make_uint2(__pack_half2(LUT[__164.x],LUT[__164.y]),__pack_half2(LUT[__164.z],LUT[__164.w]));
      uint2 __216 = make_uint2(__pack_half2(Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 12)], Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 12)]), __pack_half2(Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 12)], Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 12)]));
      ((half2*)(&(__163.x)))->x = (((half2*)(&(__215.x)))->x*((half2*)(&(__216.x)))->x);
      ((half2*)(&(__163.x)))->y = (((half2*)(&(__215.x)))->y*((half2*)(&(__216.x)))->y);
      ((half2*)(&(__163.y)))->x = (((half2*)(&(__215.y)))->x*((half2*)(&(__216.y)))->x);
      ((half2*)(&(__163.y)))->y = (((half2*)(&(__215.y)))->y*((half2*)(&(__216.y)))->y);
    *(uint2*)(B_decode_shared + ((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 4)) + 480)) = __163;
    uint2 __217;
      int4 __218;
      int __219;
        int __220;
          int4 __221;
            int4 __222 = make_int4(((((((((int)blockIdx.x) & 127) * 65536) + ((((int)threadIdx.x) >> 3) * 2048)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 32768), ((((((((int)blockIdx.x) & 127) * 65536) + ((((int)threadIdx.x) >> 3) * 2048)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 32768), ((((((((int)blockIdx.x) & 127) * 65536) + ((((int)threadIdx.x) >> 3) * 2048)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 32768), ((((((((int)blockIdx.x) & 127) * 65536) + ((((int)threadIdx.x) >> 3) * 2048)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 32768));
            int4 __223;
              int4 __224 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __225 = make_int4(2, 2, 2, 2);
              __223.x = (__224.x%__225.x);
              __223.y = (__224.y%__225.y);
              __223.z = (__224.z%__225.z);
              __223.w = (__224.w%__225.w);
            int4 __226;
              int4 __227 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __228 = make_int4(2, 2, 2, 2);
              __226.x = (__227.x/__228.x);
              __226.y = (__227.y/__228.y);
              __226.z = (__227.z/__228.z);
              __226.w = (__227.w/__228.w);
            int4 __229;
            ushort4 __230;
              ushort4 __231;
                ushort4 __232;
                  int4 __233 = make_int4(2, 2, 2, 2);
                  int4 __234 = make_int4(0, 0, 0, 0);
                  __232.x = (__233.x>=__234.x);
                  __232.y = (__233.y>=__234.y);
                  __232.z = (__233.z>=__234.z);
                  __232.w = (__233.w>=__234.w);
                ushort4 __235;
                  int4 __236 = make_int4(0, 0, 0, 0);
                  __235.x = (__223.x>=__236.x);
                  __235.y = (__223.y>=__236.y);
                  __235.z = (__223.z>=__236.z);
                  __235.w = (__223.w>=__236.w);
                __231.x = (__232.x&&__235.x);
                __231.y = (__232.y&&__235.y);
                __231.z = (__232.z&&__235.z);
                __231.w = (__232.w&&__235.w);
              ushort4 __237;
                ushort4 __238;
                  int4 __239 = make_int4(2, 2, 2, 2);
                  int4 __240 = make_int4(0, 0, 0, 0);
                  __238.x = (__239.x<__240.x);
                  __238.y = (__239.y<__240.y);
                  __238.z = (__239.z<__240.z);
                  __238.w = (__239.w<__240.w);
                ushort4 __241;
                  int4 __242 = make_int4(0, 0, 0, 0);
                  __241.x = (__223.x<=__242.x);
                  __241.y = (__223.y<=__242.y);
                  __241.z = (__223.z<=__242.z);
                  __241.w = (__223.w<=__242.w);
                __237.x = (__238.x&&__241.x);
                __237.y = (__238.y&&__241.y);
                __237.z = (__238.z&&__241.z);
                __237.w = (__238.w&&__241.w);
              __230.x = (__231.x||__237.x);
              __230.y = (__231.y||__237.y);
              __230.z = (__231.z||__237.z);
              __230.w = (__231.w||__237.w);
            int4 __243;
              int4 __244 = make_int4(1, 1, 1, 1);
              __243.x = (__226.x-__244.x);
              __243.y = (__226.y-__244.y);
              __243.z = (__226.z-__244.z);
              __243.w = (__226.w-__244.w);
            __229.x = (bool(__230.x)?__226.x:__243.x);
            __229.y = (bool(__230.y)?__226.y:__243.y);
            __229.z = (bool(__230.z)?__226.z:__243.z);
            __229.w = (bool(__230.w)?__226.w:__243.w);
            __221.x = (__222.x+__229.x);
            __221.y = (__222.y+__229.y);
            __221.z = (__222.z+__229.z);
            __221.w = (__222.w+__229.w);
          int __245 = ((0x000000ff << 0) & (B[__221.x] << 0))|((0x000000ff << 8) & (B[__221.y] << 8))|((0x000000ff << 16) & (B[__221.z] << 16))|((0x000000ff << 24) & (B[__221.w] << 24));
          int __246;
          int4 __247;
            int4 __248;
              int4 __249 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __250 = make_int4(2, 2, 2, 2);
              __248.x = (__249.x%__250.x);
              __248.y = (__249.y%__250.y);
              __248.z = (__249.z%__250.z);
              __248.w = (__249.w%__250.w);
            int4 __251;
            ushort4 __252;
              ushort4 __253;
                ushort4 __254;
                  int4 __255 = make_int4(2, 2, 2, 2);
                  int4 __256 = make_int4(0, 0, 0, 0);
                  __254.x = (__255.x>=__256.x);
                  __254.y = (__255.y>=__256.y);
                  __254.z = (__255.z>=__256.z);
                  __254.w = (__255.w>=__256.w);
                ushort4 __257;
                  int4 __258 = make_int4(0, 0, 0, 0);
                  __257.x = (__248.x>=__258.x);
                  __257.y = (__248.y>=__258.y);
                  __257.z = (__248.z>=__258.z);
                  __257.w = (__248.w>=__258.w);
                __253.x = (__254.x&&__257.x);
                __253.y = (__254.y&&__257.y);
                __253.z = (__254.z&&__257.z);
                __253.w = (__254.w&&__257.w);
              ushort4 __259;
                ushort4 __260;
                  int4 __261 = make_int4(2, 2, 2, 2);
                  int4 __262 = make_int4(0, 0, 0, 0);
                  __260.x = (__261.x<__262.x);
                  __260.y = (__261.y<__262.y);
                  __260.z = (__261.z<__262.z);
                  __260.w = (__261.w<__262.w);
                ushort4 __263;
                  int4 __264 = make_int4(0, 0, 0, 0);
                  __263.x = (__248.x<=__264.x);
                  __263.y = (__248.y<=__264.y);
                  __263.z = (__248.z<=__264.z);
                  __263.w = (__248.w<=__264.w);
                __259.x = (__260.x&&__263.x);
                __259.y = (__260.y&&__263.y);
                __259.z = (__260.z&&__263.z);
                __259.w = (__260.w&&__263.w);
              __252.x = (__253.x||__259.x);
              __252.y = (__253.y||__259.y);
              __252.z = (__253.z||__259.z);
              __252.w = (__253.w||__259.w);
            int4 __265;
              int4 __266 = make_int4(2, 2, 2, 2);
              __265.x = (__248.x+__266.x);
              __265.y = (__248.y+__266.y);
              __265.z = (__248.z+__266.z);
              __265.w = (__248.w+__266.w);
            __251.x = (bool(__252.x)?__248.x:__265.x);
            __251.y = (bool(__252.y)?__248.y:__265.y);
            __251.z = (bool(__252.z)?__248.z:__265.z);
            __251.w = (bool(__252.w)?__248.w:__265.w);
            int4 __267 = make_int4(4, 4, 4, 4);
            __247.x = (__251.x*__267.x);
            __247.y = (__251.y*__267.y);
            __247.z = (__251.z*__267.z);
            __247.w = (__251.w*__267.w);
          __246=((signed char)(__247.x) << 0);
          __246=__246 & ~(0x000000ff << 8) |((signed char)(__247.y) << 8);
          __246=__246 & ~(0x000000ff << 16) |((signed char)(__247.z) << 16);
          __246=__246 & ~(0x000000ff << 24) |((signed char)(__247.w) << 24);
          __220=((((char)(__245 >> 0)) >> ((char)(__246 >> 0))) << 0);
          __220=__220 & ~(0x000000ff << 8) |((((char)(__245 >> 8)) >> ((char)(__246 >> 8))) << 8);
          __220=__220 & ~(0x000000ff << 16) |((((char)(__245 >> 16)) >> ((char)(__246 >> 16))) << 16);
          __220=__220 & ~(0x000000ff << 24) |((((char)(__245 >> 24)) >> ((char)(__246 >> 24))) << 24);
        int __268 = (int)252645135;
        __219=((((char)(__220 >> 0)) & ((char)(__268 >> 0))) << 0);
        __219=__219 & ~(0x000000ff << 8) |((((char)(__220 >> 8)) & ((char)(__268 >> 8))) << 8);
        __219=__219 & ~(0x000000ff << 16) |((((char)(__220 >> 16)) & ((char)(__268 >> 16))) << 16);
        __219=__219 & ~(0x000000ff << 24) |((((char)(__220 >> 24)) & ((char)(__268 >> 24))) << 24);
      __218.x = (int)(((char)(__219 >> 0)));
      __218.y = (int)(((char)(__219 >> 8)));
      __218.z = (int)(((char)(__219 >> 16)));
      __218.w = (int)(((char)(__219 >> 24)));
      uint2 __269 = make_uint2(__pack_half2(LUT[__218.x],LUT[__218.y]),__pack_half2(LUT[__218.z],LUT[__218.w]));
      uint2 __270 = make_uint2(__pack_half2(Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 16)], Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 16)]), __pack_half2(Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 16)], Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 16)]));
      ((half2*)(&(__217.x)))->x = (((half2*)(&(__269.x)))->x*((half2*)(&(__270.x)))->x);
      ((half2*)(&(__217.x)))->y = (((half2*)(&(__269.x)))->y*((half2*)(&(__270.x)))->y);
      ((half2*)(&(__217.y)))->x = (((half2*)(&(__269.y)))->x*((half2*)(&(__270.y)))->x);
      ((half2*)(&(__217.y)))->y = (((half2*)(&(__269.y)))->y*((half2*)(&(__270.y)))->y);
    *(uint2*)(B_decode_shared + ((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 4)) + 640)) = __217;
    uint2 __271;
      int4 __272;
      int __273;
        int __274;
          int4 __275;
            int4 __276 = make_int4(((((((((int)blockIdx.x) & 127) * 65536) + ((((int)threadIdx.x) >> 3) * 2048)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 40960), ((((((((int)blockIdx.x) & 127) * 65536) + ((((int)threadIdx.x) >> 3) * 2048)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 40960), ((((((((int)blockIdx.x) & 127) * 65536) + ((((int)threadIdx.x) >> 3) * 2048)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 40960), ((((((((int)blockIdx.x) & 127) * 65536) + ((((int)threadIdx.x) >> 3) * 2048)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 40960));
            int4 __277;
              int4 __278 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __279 = make_int4(2, 2, 2, 2);
              __277.x = (__278.x%__279.x);
              __277.y = (__278.y%__279.y);
              __277.z = (__278.z%__279.z);
              __277.w = (__278.w%__279.w);
            int4 __280;
              int4 __281 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __282 = make_int4(2, 2, 2, 2);
              __280.x = (__281.x/__282.x);
              __280.y = (__281.y/__282.y);
              __280.z = (__281.z/__282.z);
              __280.w = (__281.w/__282.w);
            int4 __283;
            ushort4 __284;
              ushort4 __285;
                ushort4 __286;
                  int4 __287 = make_int4(2, 2, 2, 2);
                  int4 __288 = make_int4(0, 0, 0, 0);
                  __286.x = (__287.x>=__288.x);
                  __286.y = (__287.y>=__288.y);
                  __286.z = (__287.z>=__288.z);
                  __286.w = (__287.w>=__288.w);
                ushort4 __289;
                  int4 __290 = make_int4(0, 0, 0, 0);
                  __289.x = (__277.x>=__290.x);
                  __289.y = (__277.y>=__290.y);
                  __289.z = (__277.z>=__290.z);
                  __289.w = (__277.w>=__290.w);
                __285.x = (__286.x&&__289.x);
                __285.y = (__286.y&&__289.y);
                __285.z = (__286.z&&__289.z);
                __285.w = (__286.w&&__289.w);
              ushort4 __291;
                ushort4 __292;
                  int4 __293 = make_int4(2, 2, 2, 2);
                  int4 __294 = make_int4(0, 0, 0, 0);
                  __292.x = (__293.x<__294.x);
                  __292.y = (__293.y<__294.y);
                  __292.z = (__293.z<__294.z);
                  __292.w = (__293.w<__294.w);
                ushort4 __295;
                  int4 __296 = make_int4(0, 0, 0, 0);
                  __295.x = (__277.x<=__296.x);
                  __295.y = (__277.y<=__296.y);
                  __295.z = (__277.z<=__296.z);
                  __295.w = (__277.w<=__296.w);
                __291.x = (__292.x&&__295.x);
                __291.y = (__292.y&&__295.y);
                __291.z = (__292.z&&__295.z);
                __291.w = (__292.w&&__295.w);
              __284.x = (__285.x||__291.x);
              __284.y = (__285.y||__291.y);
              __284.z = (__285.z||__291.z);
              __284.w = (__285.w||__291.w);
            int4 __297;
              int4 __298 = make_int4(1, 1, 1, 1);
              __297.x = (__280.x-__298.x);
              __297.y = (__280.y-__298.y);
              __297.z = (__280.z-__298.z);
              __297.w = (__280.w-__298.w);
            __283.x = (bool(__284.x)?__280.x:__297.x);
            __283.y = (bool(__284.y)?__280.y:__297.y);
            __283.z = (bool(__284.z)?__280.z:__297.z);
            __283.w = (bool(__284.w)?__280.w:__297.w);
            __275.x = (__276.x+__283.x);
            __275.y = (__276.y+__283.y);
            __275.z = (__276.z+__283.z);
            __275.w = (__276.w+__283.w);
          int __299 = ((0x000000ff << 0) & (B[__275.x] << 0))|((0x000000ff << 8) & (B[__275.y] << 8))|((0x000000ff << 16) & (B[__275.z] << 16))|((0x000000ff << 24) & (B[__275.w] << 24));
          int __300;
          int4 __301;
            int4 __302;
              int4 __303 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __304 = make_int4(2, 2, 2, 2);
              __302.x = (__303.x%__304.x);
              __302.y = (__303.y%__304.y);
              __302.z = (__303.z%__304.z);
              __302.w = (__303.w%__304.w);
            int4 __305;
            ushort4 __306;
              ushort4 __307;
                ushort4 __308;
                  int4 __309 = make_int4(2, 2, 2, 2);
                  int4 __310 = make_int4(0, 0, 0, 0);
                  __308.x = (__309.x>=__310.x);
                  __308.y = (__309.y>=__310.y);
                  __308.z = (__309.z>=__310.z);
                  __308.w = (__309.w>=__310.w);
                ushort4 __311;
                  int4 __312 = make_int4(0, 0, 0, 0);
                  __311.x = (__302.x>=__312.x);
                  __311.y = (__302.y>=__312.y);
                  __311.z = (__302.z>=__312.z);
                  __311.w = (__302.w>=__312.w);
                __307.x = (__308.x&&__311.x);
                __307.y = (__308.y&&__311.y);
                __307.z = (__308.z&&__311.z);
                __307.w = (__308.w&&__311.w);
              ushort4 __313;
                ushort4 __314;
                  int4 __315 = make_int4(2, 2, 2, 2);
                  int4 __316 = make_int4(0, 0, 0, 0);
                  __314.x = (__315.x<__316.x);
                  __314.y = (__315.y<__316.y);
                  __314.z = (__315.z<__316.z);
                  __314.w = (__315.w<__316.w);
                ushort4 __317;
                  int4 __318 = make_int4(0, 0, 0, 0);
                  __317.x = (__302.x<=__318.x);
                  __317.y = (__302.y<=__318.y);
                  __317.z = (__302.z<=__318.z);
                  __317.w = (__302.w<=__318.w);
                __313.x = (__314.x&&__317.x);
                __313.y = (__314.y&&__317.y);
                __313.z = (__314.z&&__317.z);
                __313.w = (__314.w&&__317.w);
              __306.x = (__307.x||__313.x);
              __306.y = (__307.y||__313.y);
              __306.z = (__307.z||__313.z);
              __306.w = (__307.w||__313.w);
            int4 __319;
              int4 __320 = make_int4(2, 2, 2, 2);
              __319.x = (__302.x+__320.x);
              __319.y = (__302.y+__320.y);
              __319.z = (__302.z+__320.z);
              __319.w = (__302.w+__320.w);
            __305.x = (bool(__306.x)?__302.x:__319.x);
            __305.y = (bool(__306.y)?__302.y:__319.y);
            __305.z = (bool(__306.z)?__302.z:__319.z);
            __305.w = (bool(__306.w)?__302.w:__319.w);
            int4 __321 = make_int4(4, 4, 4, 4);
            __301.x = (__305.x*__321.x);
            __301.y = (__305.y*__321.y);
            __301.z = (__305.z*__321.z);
            __301.w = (__305.w*__321.w);
          __300=((signed char)(__301.x) << 0);
          __300=__300 & ~(0x000000ff << 8) |((signed char)(__301.y) << 8);
          __300=__300 & ~(0x000000ff << 16) |((signed char)(__301.z) << 16);
          __300=__300 & ~(0x000000ff << 24) |((signed char)(__301.w) << 24);
          __274=((((char)(__299 >> 0)) >> ((char)(__300 >> 0))) << 0);
          __274=__274 & ~(0x000000ff << 8) |((((char)(__299 >> 8)) >> ((char)(__300 >> 8))) << 8);
          __274=__274 & ~(0x000000ff << 16) |((((char)(__299 >> 16)) >> ((char)(__300 >> 16))) << 16);
          __274=__274 & ~(0x000000ff << 24) |((((char)(__299 >> 24)) >> ((char)(__300 >> 24))) << 24);
        int __322 = (int)252645135;
        __273=((((char)(__274 >> 0)) & ((char)(__322 >> 0))) << 0);
        __273=__273 & ~(0x000000ff << 8) |((((char)(__274 >> 8)) & ((char)(__322 >> 8))) << 8);
        __273=__273 & ~(0x000000ff << 16) |((((char)(__274 >> 16)) & ((char)(__322 >> 16))) << 16);
        __273=__273 & ~(0x000000ff << 24) |((((char)(__274 >> 24)) & ((char)(__322 >> 24))) << 24);
      __272.x = (int)(((char)(__273 >> 0)));
      __272.y = (int)(((char)(__273 >> 8)));
      __272.z = (int)(((char)(__273 >> 16)));
      __272.w = (int)(((char)(__273 >> 24)));
      uint2 __323 = make_uint2(__pack_half2(LUT[__272.x],LUT[__272.y]),__pack_half2(LUT[__272.z],LUT[__272.w]));
      uint2 __324 = make_uint2(__pack_half2(Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 20)], Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 20)]), __pack_half2(Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 20)], Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 20)]));
      ((half2*)(&(__271.x)))->x = (((half2*)(&(__323.x)))->x*((half2*)(&(__324.x)))->x);
      ((half2*)(&(__271.x)))->y = (((half2*)(&(__323.x)))->y*((half2*)(&(__324.x)))->y);
      ((half2*)(&(__271.y)))->x = (((half2*)(&(__323.y)))->x*((half2*)(&(__324.y)))->x);
      ((half2*)(&(__271.y)))->y = (((half2*)(&(__323.y)))->y*((half2*)(&(__324.y)))->y);
    *(uint2*)(B_decode_shared + ((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 4)) + 800)) = __271;
    uint2 __325;
      int4 __326;
      int __327;
        int __328;
          int4 __329;
            int4 __330 = make_int4(((((((((int)blockIdx.x) & 127) * 65536) + ((((int)threadIdx.x) >> 3) * 2048)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 49152), ((((((((int)blockIdx.x) & 127) * 65536) + ((((int)threadIdx.x) >> 3) * 2048)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 49152), ((((((((int)blockIdx.x) & 127) * 65536) + ((((int)threadIdx.x) >> 3) * 2048)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 49152), ((((((((int)blockIdx.x) & 127) * 65536) + ((((int)threadIdx.x) >> 3) * 2048)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 49152));
            int4 __331;
              int4 __332 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __333 = make_int4(2, 2, 2, 2);
              __331.x = (__332.x%__333.x);
              __331.y = (__332.y%__333.y);
              __331.z = (__332.z%__333.z);
              __331.w = (__332.w%__333.w);
            int4 __334;
              int4 __335 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __336 = make_int4(2, 2, 2, 2);
              __334.x = (__335.x/__336.x);
              __334.y = (__335.y/__336.y);
              __334.z = (__335.z/__336.z);
              __334.w = (__335.w/__336.w);
            int4 __337;
            ushort4 __338;
              ushort4 __339;
                ushort4 __340;
                  int4 __341 = make_int4(2, 2, 2, 2);
                  int4 __342 = make_int4(0, 0, 0, 0);
                  __340.x = (__341.x>=__342.x);
                  __340.y = (__341.y>=__342.y);
                  __340.z = (__341.z>=__342.z);
                  __340.w = (__341.w>=__342.w);
                ushort4 __343;
                  int4 __344 = make_int4(0, 0, 0, 0);
                  __343.x = (__331.x>=__344.x);
                  __343.y = (__331.y>=__344.y);
                  __343.z = (__331.z>=__344.z);
                  __343.w = (__331.w>=__344.w);
                __339.x = (__340.x&&__343.x);
                __339.y = (__340.y&&__343.y);
                __339.z = (__340.z&&__343.z);
                __339.w = (__340.w&&__343.w);
              ushort4 __345;
                ushort4 __346;
                  int4 __347 = make_int4(2, 2, 2, 2);
                  int4 __348 = make_int4(0, 0, 0, 0);
                  __346.x = (__347.x<__348.x);
                  __346.y = (__347.y<__348.y);
                  __346.z = (__347.z<__348.z);
                  __346.w = (__347.w<__348.w);
                ushort4 __349;
                  int4 __350 = make_int4(0, 0, 0, 0);
                  __349.x = (__331.x<=__350.x);
                  __349.y = (__331.y<=__350.y);
                  __349.z = (__331.z<=__350.z);
                  __349.w = (__331.w<=__350.w);
                __345.x = (__346.x&&__349.x);
                __345.y = (__346.y&&__349.y);
                __345.z = (__346.z&&__349.z);
                __345.w = (__346.w&&__349.w);
              __338.x = (__339.x||__345.x);
              __338.y = (__339.y||__345.y);
              __338.z = (__339.z||__345.z);
              __338.w = (__339.w||__345.w);
            int4 __351;
              int4 __352 = make_int4(1, 1, 1, 1);
              __351.x = (__334.x-__352.x);
              __351.y = (__334.y-__352.y);
              __351.z = (__334.z-__352.z);
              __351.w = (__334.w-__352.w);
            __337.x = (bool(__338.x)?__334.x:__351.x);
            __337.y = (bool(__338.y)?__334.y:__351.y);
            __337.z = (bool(__338.z)?__334.z:__351.z);
            __337.w = (bool(__338.w)?__334.w:__351.w);
            __329.x = (__330.x+__337.x);
            __329.y = (__330.y+__337.y);
            __329.z = (__330.z+__337.z);
            __329.w = (__330.w+__337.w);
          int __353 = ((0x000000ff << 0) & (B[__329.x] << 0))|((0x000000ff << 8) & (B[__329.y] << 8))|((0x000000ff << 16) & (B[__329.z] << 16))|((0x000000ff << 24) & (B[__329.w] << 24));
          int __354;
          int4 __355;
            int4 __356;
              int4 __357 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __358 = make_int4(2, 2, 2, 2);
              __356.x = (__357.x%__358.x);
              __356.y = (__357.y%__358.y);
              __356.z = (__357.z%__358.z);
              __356.w = (__357.w%__358.w);
            int4 __359;
            ushort4 __360;
              ushort4 __361;
                ushort4 __362;
                  int4 __363 = make_int4(2, 2, 2, 2);
                  int4 __364 = make_int4(0, 0, 0, 0);
                  __362.x = (__363.x>=__364.x);
                  __362.y = (__363.y>=__364.y);
                  __362.z = (__363.z>=__364.z);
                  __362.w = (__363.w>=__364.w);
                ushort4 __365;
                  int4 __366 = make_int4(0, 0, 0, 0);
                  __365.x = (__356.x>=__366.x);
                  __365.y = (__356.y>=__366.y);
                  __365.z = (__356.z>=__366.z);
                  __365.w = (__356.w>=__366.w);
                __361.x = (__362.x&&__365.x);
                __361.y = (__362.y&&__365.y);
                __361.z = (__362.z&&__365.z);
                __361.w = (__362.w&&__365.w);
              ushort4 __367;
                ushort4 __368;
                  int4 __369 = make_int4(2, 2, 2, 2);
                  int4 __370 = make_int4(0, 0, 0, 0);
                  __368.x = (__369.x<__370.x);
                  __368.y = (__369.y<__370.y);
                  __368.z = (__369.z<__370.z);
                  __368.w = (__369.w<__370.w);
                ushort4 __371;
                  int4 __372 = make_int4(0, 0, 0, 0);
                  __371.x = (__356.x<=__372.x);
                  __371.y = (__356.y<=__372.y);
                  __371.z = (__356.z<=__372.z);
                  __371.w = (__356.w<=__372.w);
                __367.x = (__368.x&&__371.x);
                __367.y = (__368.y&&__371.y);
                __367.z = (__368.z&&__371.z);
                __367.w = (__368.w&&__371.w);
              __360.x = (__361.x||__367.x);
              __360.y = (__361.y||__367.y);
              __360.z = (__361.z||__367.z);
              __360.w = (__361.w||__367.w);
            int4 __373;
              int4 __374 = make_int4(2, 2, 2, 2);
              __373.x = (__356.x+__374.x);
              __373.y = (__356.y+__374.y);
              __373.z = (__356.z+__374.z);
              __373.w = (__356.w+__374.w);
            __359.x = (bool(__360.x)?__356.x:__373.x);
            __359.y = (bool(__360.y)?__356.y:__373.y);
            __359.z = (bool(__360.z)?__356.z:__373.z);
            __359.w = (bool(__360.w)?__356.w:__373.w);
            int4 __375 = make_int4(4, 4, 4, 4);
            __355.x = (__359.x*__375.x);
            __355.y = (__359.y*__375.y);
            __355.z = (__359.z*__375.z);
            __355.w = (__359.w*__375.w);
          __354=((signed char)(__355.x) << 0);
          __354=__354 & ~(0x000000ff << 8) |((signed char)(__355.y) << 8);
          __354=__354 & ~(0x000000ff << 16) |((signed char)(__355.z) << 16);
          __354=__354 & ~(0x000000ff << 24) |((signed char)(__355.w) << 24);
          __328=((((char)(__353 >> 0)) >> ((char)(__354 >> 0))) << 0);
          __328=__328 & ~(0x000000ff << 8) |((((char)(__353 >> 8)) >> ((char)(__354 >> 8))) << 8);
          __328=__328 & ~(0x000000ff << 16) |((((char)(__353 >> 16)) >> ((char)(__354 >> 16))) << 16);
          __328=__328 & ~(0x000000ff << 24) |((((char)(__353 >> 24)) >> ((char)(__354 >> 24))) << 24);
        int __376 = (int)252645135;
        __327=((((char)(__328 >> 0)) & ((char)(__376 >> 0))) << 0);
        __327=__327 & ~(0x000000ff << 8) |((((char)(__328 >> 8)) & ((char)(__376 >> 8))) << 8);
        __327=__327 & ~(0x000000ff << 16) |((((char)(__328 >> 16)) & ((char)(__376 >> 16))) << 16);
        __327=__327 & ~(0x000000ff << 24) |((((char)(__328 >> 24)) & ((char)(__376 >> 24))) << 24);
      __326.x = (int)(((char)(__327 >> 0)));
      __326.y = (int)(((char)(__327 >> 8)));
      __326.z = (int)(((char)(__327 >> 16)));
      __326.w = (int)(((char)(__327 >> 24)));
      uint2 __377 = make_uint2(__pack_half2(LUT[__326.x],LUT[__326.y]),__pack_half2(LUT[__326.z],LUT[__326.w]));
      uint2 __378 = make_uint2(__pack_half2(Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 24)], Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 24)]), __pack_half2(Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 24)], Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 24)]));
      ((half2*)(&(__325.x)))->x = (((half2*)(&(__377.x)))->x*((half2*)(&(__378.x)))->x);
      ((half2*)(&(__325.x)))->y = (((half2*)(&(__377.x)))->y*((half2*)(&(__378.x)))->y);
      ((half2*)(&(__325.y)))->x = (((half2*)(&(__377.y)))->x*((half2*)(&(__378.y)))->x);
      ((half2*)(&(__325.y)))->y = (((half2*)(&(__377.y)))->y*((half2*)(&(__378.y)))->y);
    *(uint2*)(B_decode_shared + ((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 4)) + 960)) = __325;
    uint2 __379;
      int4 __380;
      int __381;
        int __382;
          int4 __383;
            int4 __384 = make_int4(((((((((int)blockIdx.x) & 127) * 65536) + ((((int)threadIdx.x) >> 3) * 2048)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 57344), ((((((((int)blockIdx.x) & 127) * 65536) + ((((int)threadIdx.x) >> 3) * 2048)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 57344), ((((((((int)blockIdx.x) & 127) * 65536) + ((((int)threadIdx.x) >> 3) * 2048)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 57344), ((((((((int)blockIdx.x) & 127) * 65536) + ((((int)threadIdx.x) >> 3) * 2048)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 57344));
            int4 __385;
              int4 __386 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __387 = make_int4(2, 2, 2, 2);
              __385.x = (__386.x%__387.x);
              __385.y = (__386.y%__387.y);
              __385.z = (__386.z%__387.z);
              __385.w = (__386.w%__387.w);
            int4 __388;
              int4 __389 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __390 = make_int4(2, 2, 2, 2);
              __388.x = (__389.x/__390.x);
              __388.y = (__389.y/__390.y);
              __388.z = (__389.z/__390.z);
              __388.w = (__389.w/__390.w);
            int4 __391;
            ushort4 __392;
              ushort4 __393;
                ushort4 __394;
                  int4 __395 = make_int4(2, 2, 2, 2);
                  int4 __396 = make_int4(0, 0, 0, 0);
                  __394.x = (__395.x>=__396.x);
                  __394.y = (__395.y>=__396.y);
                  __394.z = (__395.z>=__396.z);
                  __394.w = (__395.w>=__396.w);
                ushort4 __397;
                  int4 __398 = make_int4(0, 0, 0, 0);
                  __397.x = (__385.x>=__398.x);
                  __397.y = (__385.y>=__398.y);
                  __397.z = (__385.z>=__398.z);
                  __397.w = (__385.w>=__398.w);
                __393.x = (__394.x&&__397.x);
                __393.y = (__394.y&&__397.y);
                __393.z = (__394.z&&__397.z);
                __393.w = (__394.w&&__397.w);
              ushort4 __399;
                ushort4 __400;
                  int4 __401 = make_int4(2, 2, 2, 2);
                  int4 __402 = make_int4(0, 0, 0, 0);
                  __400.x = (__401.x<__402.x);
                  __400.y = (__401.y<__402.y);
                  __400.z = (__401.z<__402.z);
                  __400.w = (__401.w<__402.w);
                ushort4 __403;
                  int4 __404 = make_int4(0, 0, 0, 0);
                  __403.x = (__385.x<=__404.x);
                  __403.y = (__385.y<=__404.y);
                  __403.z = (__385.z<=__404.z);
                  __403.w = (__385.w<=__404.w);
                __399.x = (__400.x&&__403.x);
                __399.y = (__400.y&&__403.y);
                __399.z = (__400.z&&__403.z);
                __399.w = (__400.w&&__403.w);
              __392.x = (__393.x||__399.x);
              __392.y = (__393.y||__399.y);
              __392.z = (__393.z||__399.z);
              __392.w = (__393.w||__399.w);
            int4 __405;
              int4 __406 = make_int4(1, 1, 1, 1);
              __405.x = (__388.x-__406.x);
              __405.y = (__388.y-__406.y);
              __405.z = (__388.z-__406.z);
              __405.w = (__388.w-__406.w);
            __391.x = (bool(__392.x)?__388.x:__405.x);
            __391.y = (bool(__392.y)?__388.y:__405.y);
            __391.z = (bool(__392.z)?__388.z:__405.z);
            __391.w = (bool(__392.w)?__388.w:__405.w);
            __383.x = (__384.x+__391.x);
            __383.y = (__384.y+__391.y);
            __383.z = (__384.z+__391.z);
            __383.w = (__384.w+__391.w);
          int __407 = ((0x000000ff << 0) & (B[__383.x] << 0))|((0x000000ff << 8) & (B[__383.y] << 8))|((0x000000ff << 16) & (B[__383.z] << 16))|((0x000000ff << 24) & (B[__383.w] << 24));
          int __408;
          int4 __409;
            int4 __410;
              int4 __411 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __412 = make_int4(2, 2, 2, 2);
              __410.x = (__411.x%__412.x);
              __410.y = (__411.y%__412.y);
              __410.z = (__411.z%__412.z);
              __410.w = (__411.w%__412.w);
            int4 __413;
            ushort4 __414;
              ushort4 __415;
                ushort4 __416;
                  int4 __417 = make_int4(2, 2, 2, 2);
                  int4 __418 = make_int4(0, 0, 0, 0);
                  __416.x = (__417.x>=__418.x);
                  __416.y = (__417.y>=__418.y);
                  __416.z = (__417.z>=__418.z);
                  __416.w = (__417.w>=__418.w);
                ushort4 __419;
                  int4 __420 = make_int4(0, 0, 0, 0);
                  __419.x = (__410.x>=__420.x);
                  __419.y = (__410.y>=__420.y);
                  __419.z = (__410.z>=__420.z);
                  __419.w = (__410.w>=__420.w);
                __415.x = (__416.x&&__419.x);
                __415.y = (__416.y&&__419.y);
                __415.z = (__416.z&&__419.z);
                __415.w = (__416.w&&__419.w);
              ushort4 __421;
                ushort4 __422;
                  int4 __423 = make_int4(2, 2, 2, 2);
                  int4 __424 = make_int4(0, 0, 0, 0);
                  __422.x = (__423.x<__424.x);
                  __422.y = (__423.y<__424.y);
                  __422.z = (__423.z<__424.z);
                  __422.w = (__423.w<__424.w);
                ushort4 __425;
                  int4 __426 = make_int4(0, 0, 0, 0);
                  __425.x = (__410.x<=__426.x);
                  __425.y = (__410.y<=__426.y);
                  __425.z = (__410.z<=__426.z);
                  __425.w = (__410.w<=__426.w);
                __421.x = (__422.x&&__425.x);
                __421.y = (__422.y&&__425.y);
                __421.z = (__422.z&&__425.z);
                __421.w = (__422.w&&__425.w);
              __414.x = (__415.x||__421.x);
              __414.y = (__415.y||__421.y);
              __414.z = (__415.z||__421.z);
              __414.w = (__415.w||__421.w);
            int4 __427;
              int4 __428 = make_int4(2, 2, 2, 2);
              __427.x = (__410.x+__428.x);
              __427.y = (__410.y+__428.y);
              __427.z = (__410.z+__428.z);
              __427.w = (__410.w+__428.w);
            __413.x = (bool(__414.x)?__410.x:__427.x);
            __413.y = (bool(__414.y)?__410.y:__427.y);
            __413.z = (bool(__414.z)?__410.z:__427.z);
            __413.w = (bool(__414.w)?__410.w:__427.w);
            int4 __429 = make_int4(4, 4, 4, 4);
            __409.x = (__413.x*__429.x);
            __409.y = (__413.y*__429.y);
            __409.z = (__413.z*__429.z);
            __409.w = (__413.w*__429.w);
          __408=((signed char)(__409.x) << 0);
          __408=__408 & ~(0x000000ff << 8) |((signed char)(__409.y) << 8);
          __408=__408 & ~(0x000000ff << 16) |((signed char)(__409.z) << 16);
          __408=__408 & ~(0x000000ff << 24) |((signed char)(__409.w) << 24);
          __382=((((char)(__407 >> 0)) >> ((char)(__408 >> 0))) << 0);
          __382=__382 & ~(0x000000ff << 8) |((((char)(__407 >> 8)) >> ((char)(__408 >> 8))) << 8);
          __382=__382 & ~(0x000000ff << 16) |((((char)(__407 >> 16)) >> ((char)(__408 >> 16))) << 16);
          __382=__382 & ~(0x000000ff << 24) |((((char)(__407 >> 24)) >> ((char)(__408 >> 24))) << 24);
        int __430 = (int)252645135;
        __381=((((char)(__382 >> 0)) & ((char)(__430 >> 0))) << 0);
        __381=__381 & ~(0x000000ff << 8) |((((char)(__382 >> 8)) & ((char)(__430 >> 8))) << 8);
        __381=__381 & ~(0x000000ff << 16) |((((char)(__382 >> 16)) & ((char)(__430 >> 16))) << 16);
        __381=__381 & ~(0x000000ff << 24) |((((char)(__382 >> 24)) & ((char)(__430 >> 24))) << 24);
      __380.x = (int)(((char)(__381 >> 0)));
      __380.y = (int)(((char)(__381 >> 8)));
      __380.z = (int)(((char)(__381 >> 16)));
      __380.w = (int)(((char)(__381 >> 24)));
      uint2 __431 = make_uint2(__pack_half2(LUT[__380.x],LUT[__380.y]),__pack_half2(LUT[__380.z],LUT[__380.w]));
      uint2 __432 = make_uint2(__pack_half2(Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 28)], Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 28)]), __pack_half2(Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 28)], Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 28)]));
      ((half2*)(&(__379.x)))->x = (((half2*)(&(__431.x)))->x*((half2*)(&(__432.x)))->x);
      ((half2*)(&(__379.x)))->y = (((half2*)(&(__431.x)))->y*((half2*)(&(__432.x)))->y);
      ((half2*)(&(__379.y)))->x = (((half2*)(&(__431.y)))->x*((half2*)(&(__432.y)))->x);
      ((half2*)(&(__379.y)))->y = (((half2*)(&(__431.y)))->y*((half2*)(&(__432.y)))->y);
    *(uint2*)(B_decode_shared + ((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 4)) + 1120)) = __379;
    __syncthreads();
    for (int k_inner_outer = 0; k_inner_outer < 2; ++k_inner_outer) {
      nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[0], (&(A_shared[(k_inner_outer * 16)])), 40);
      nvcuda::wmma::load_matrix_sync(B_decode_shared_wmma_matrix_b[0], (&(B_decode_shared[(k_inner_outer * 16)])), 40);
      nvcuda::wmma::mma_sync(C_wmma_accumulator[0], A_shared_wmma_matrix_a[0], B_decode_shared_wmma_matrix_b[0], C_wmma_accumulator[0]);
    }
  }
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(C[(((((int)blockIdx.x) >> 7) * 32768) + ((((int)blockIdx.x) & 127) * 32))])), C_wmma_accumulator[0], 4096, nvcuda::wmma::mem_row_major);
  __syncthreads();
}



__global__ void __launch_bounds__(32) cutlass_kernel_fp16_nf4_fp16_m16n4096k11008_nt_8x32x32_8x32x16_(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ C) {
            int8_t* B = ((int8_t *)QB);
	 half* LUT = (half *)((int8_t *)QB + 22544384);
	 half* Scales = (half *)((int8_t *)QB + 22544384 + 32);                 
            // const dim3 GridDim(256, 1, 1);
            // const dim3 BlockDim(32, 1, 1);
            // cutlass_kernel_fp16_nf4_fp16_m16n4096k11008_nt_8x32x32_8x32x16_<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
  
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, half> C_wmma_accumulator[1];
  __shared__ half A_shared[320];
  __shared__ half B_decode_shared[1280];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, half, nvcuda::wmma::row_major> A_shared_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, half, nvcuda::wmma::col_major> B_decode_shared_wmma_matrix_b[1];
  nvcuda::wmma::fill_fragment(C_wmma_accumulator[0], __float2half_rn(0.000000e+00f));
  for (int k_outer = 0; k_outer < 344; ++k_outer) {
    __syncthreads();
    *(uint4*)(A_shared + (((((int)threadIdx.x) >> 2) * 40) + ((((int)threadIdx.x) & 3) * 8))) = *(uint4*)(A + (((((((int)blockIdx.x) >> 7) * 88064) + ((((int)threadIdx.x) >> 2) * 11008)) + (k_outer * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    uint2 __1;
      int4 __2;
      int __3;
        int __4;
          int4 __5;
            int4 __6 = make_int4((((((((int)blockIdx.x) & 127) * 176128) + ((((int)threadIdx.x) >> 3) * 5504)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)), (((((((int)blockIdx.x) & 127) * 176128) + ((((int)threadIdx.x) >> 3) * 5504)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)), (((((((int)blockIdx.x) & 127) * 176128) + ((((int)threadIdx.x) >> 3) * 5504)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)), (((((((int)blockIdx.x) & 127) * 176128) + ((((int)threadIdx.x) >> 3) * 5504)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)));
            int4 __7;
              int4 __8 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __9 = make_int4(2, 2, 2, 2);
              __7.x = (__8.x%__9.x);
              __7.y = (__8.y%__9.y);
              __7.z = (__8.z%__9.z);
              __7.w = (__8.w%__9.w);
            int4 __10;
              int4 __11 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __12 = make_int4(2, 2, 2, 2);
              __10.x = (__11.x/__12.x);
              __10.y = (__11.y/__12.y);
              __10.z = (__11.z/__12.z);
              __10.w = (__11.w/__12.w);
            int4 __13;
            ushort4 __14;
              ushort4 __15;
                ushort4 __16;
                  int4 __17 = make_int4(2, 2, 2, 2);
                  int4 __18 = make_int4(0, 0, 0, 0);
                  __16.x = (__17.x>=__18.x);
                  __16.y = (__17.y>=__18.y);
                  __16.z = (__17.z>=__18.z);
                  __16.w = (__17.w>=__18.w);
                ushort4 __19;
                  int4 __20 = make_int4(0, 0, 0, 0);
                  __19.x = (__7.x>=__20.x);
                  __19.y = (__7.y>=__20.y);
                  __19.z = (__7.z>=__20.z);
                  __19.w = (__7.w>=__20.w);
                __15.x = (__16.x&&__19.x);
                __15.y = (__16.y&&__19.y);
                __15.z = (__16.z&&__19.z);
                __15.w = (__16.w&&__19.w);
              ushort4 __21;
                ushort4 __22;
                  int4 __23 = make_int4(2, 2, 2, 2);
                  int4 __24 = make_int4(0, 0, 0, 0);
                  __22.x = (__23.x<__24.x);
                  __22.y = (__23.y<__24.y);
                  __22.z = (__23.z<__24.z);
                  __22.w = (__23.w<__24.w);
                ushort4 __25;
                  int4 __26 = make_int4(0, 0, 0, 0);
                  __25.x = (__7.x<=__26.x);
                  __25.y = (__7.y<=__26.y);
                  __25.z = (__7.z<=__26.z);
                  __25.w = (__7.w<=__26.w);
                __21.x = (__22.x&&__25.x);
                __21.y = (__22.y&&__25.y);
                __21.z = (__22.z&&__25.z);
                __21.w = (__22.w&&__25.w);
              __14.x = (__15.x||__21.x);
              __14.y = (__15.y||__21.y);
              __14.z = (__15.z||__21.z);
              __14.w = (__15.w||__21.w);
            int4 __27;
              int4 __28 = make_int4(1, 1, 1, 1);
              __27.x = (__10.x-__28.x);
              __27.y = (__10.y-__28.y);
              __27.z = (__10.z-__28.z);
              __27.w = (__10.w-__28.w);
            __13.x = (bool(__14.x)?__10.x:__27.x);
            __13.y = (bool(__14.y)?__10.y:__27.y);
            __13.z = (bool(__14.z)?__10.z:__27.z);
            __13.w = (bool(__14.w)?__10.w:__27.w);
            __5.x = (__6.x+__13.x);
            __5.y = (__6.y+__13.y);
            __5.z = (__6.z+__13.z);
            __5.w = (__6.w+__13.w);
          int __29 = ((0x000000ff << 0) & (B[__5.x] << 0))|((0x000000ff << 8) & (B[__5.y] << 8))|((0x000000ff << 16) & (B[__5.z] << 16))|((0x000000ff << 24) & (B[__5.w] << 24));
          int __30;
          int4 __31;
            int4 __32;
              int4 __33 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __34 = make_int4(2, 2, 2, 2);
              __32.x = (__33.x%__34.x);
              __32.y = (__33.y%__34.y);
              __32.z = (__33.z%__34.z);
              __32.w = (__33.w%__34.w);
            int4 __35;
            ushort4 __36;
              ushort4 __37;
                ushort4 __38;
                  int4 __39 = make_int4(2, 2, 2, 2);
                  int4 __40 = make_int4(0, 0, 0, 0);
                  __38.x = (__39.x>=__40.x);
                  __38.y = (__39.y>=__40.y);
                  __38.z = (__39.z>=__40.z);
                  __38.w = (__39.w>=__40.w);
                ushort4 __41;
                  int4 __42 = make_int4(0, 0, 0, 0);
                  __41.x = (__32.x>=__42.x);
                  __41.y = (__32.y>=__42.y);
                  __41.z = (__32.z>=__42.z);
                  __41.w = (__32.w>=__42.w);
                __37.x = (__38.x&&__41.x);
                __37.y = (__38.y&&__41.y);
                __37.z = (__38.z&&__41.z);
                __37.w = (__38.w&&__41.w);
              ushort4 __43;
                ushort4 __44;
                  int4 __45 = make_int4(2, 2, 2, 2);
                  int4 __46 = make_int4(0, 0, 0, 0);
                  __44.x = (__45.x<__46.x);
                  __44.y = (__45.y<__46.y);
                  __44.z = (__45.z<__46.z);
                  __44.w = (__45.w<__46.w);
                ushort4 __47;
                  int4 __48 = make_int4(0, 0, 0, 0);
                  __47.x = (__32.x<=__48.x);
                  __47.y = (__32.y<=__48.y);
                  __47.z = (__32.z<=__48.z);
                  __47.w = (__32.w<=__48.w);
                __43.x = (__44.x&&__47.x);
                __43.y = (__44.y&&__47.y);
                __43.z = (__44.z&&__47.z);
                __43.w = (__44.w&&__47.w);
              __36.x = (__37.x||__43.x);
              __36.y = (__37.y||__43.y);
              __36.z = (__37.z||__43.z);
              __36.w = (__37.w||__43.w);
            int4 __49;
              int4 __50 = make_int4(2, 2, 2, 2);
              __49.x = (__32.x+__50.x);
              __49.y = (__32.y+__50.y);
              __49.z = (__32.z+__50.z);
              __49.w = (__32.w+__50.w);
            __35.x = (bool(__36.x)?__32.x:__49.x);
            __35.y = (bool(__36.y)?__32.y:__49.y);
            __35.z = (bool(__36.z)?__32.z:__49.z);
            __35.w = (bool(__36.w)?__32.w:__49.w);
            int4 __51 = make_int4(4, 4, 4, 4);
            __31.x = (__35.x*__51.x);
            __31.y = (__35.y*__51.y);
            __31.z = (__35.z*__51.z);
            __31.w = (__35.w*__51.w);
          __30=((signed char)(__31.x) << 0);
          __30=__30 & ~(0x000000ff << 8) |((signed char)(__31.y) << 8);
          __30=__30 & ~(0x000000ff << 16) |((signed char)(__31.z) << 16);
          __30=__30 & ~(0x000000ff << 24) |((signed char)(__31.w) << 24);
          __4=((((char)(__29 >> 0)) >> ((char)(__30 >> 0))) << 0);
          __4=__4 & ~(0x000000ff << 8) |((((char)(__29 >> 8)) >> ((char)(__30 >> 8))) << 8);
          __4=__4 & ~(0x000000ff << 16) |((((char)(__29 >> 16)) >> ((char)(__30 >> 16))) << 16);
          __4=__4 & ~(0x000000ff << 24) |((((char)(__29 >> 24)) >> ((char)(__30 >> 24))) << 24);
        int __52 = (int)252645135;
        __3=((((char)(__4 >> 0)) & ((char)(__52 >> 0))) << 0);
        __3=__3 & ~(0x000000ff << 8) |((((char)(__4 >> 8)) & ((char)(__52 >> 8))) << 8);
        __3=__3 & ~(0x000000ff << 16) |((((char)(__4 >> 16)) & ((char)(__52 >> 16))) << 16);
        __3=__3 & ~(0x000000ff << 24) |((((char)(__4 >> 24)) & ((char)(__52 >> 24))) << 24);
      __2.x = (int)(((char)(__3 >> 0)));
      __2.y = (int)(((char)(__3 >> 8)));
      __2.z = (int)(((char)(__3 >> 16)));
      __2.w = (int)(((char)(__3 >> 24)));
      uint2 __53 = make_uint2(__pack_half2(LUT[__2.x],LUT[__2.y]),__pack_half2(LUT[__2.z],LUT[__2.w]));
      uint2 __54 = make_uint2(__pack_half2(Scales[((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3))], Scales[((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3))]), __pack_half2(Scales[((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3))], Scales[((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3))]));
      ((half2*)(&(__1.x)))->x = (((half2*)(&(__53.x)))->x*((half2*)(&(__54.x)))->x);
      ((half2*)(&(__1.x)))->y = (((half2*)(&(__53.x)))->y*((half2*)(&(__54.x)))->y);
      ((half2*)(&(__1.y)))->x = (((half2*)(&(__53.y)))->x*((half2*)(&(__54.y)))->x);
      ((half2*)(&(__1.y)))->y = (((half2*)(&(__53.y)))->y*((half2*)(&(__54.y)))->y);
    *(uint2*)(B_decode_shared + (((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 4))) = __1;
    uint2 __55;
      int4 __56;
      int __57;
        int __58;
          int4 __59;
            int4 __60 = make_int4(((((((((int)blockIdx.x) & 127) * 176128) + ((((int)threadIdx.x) >> 3) * 5504)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 22016), ((((((((int)blockIdx.x) & 127) * 176128) + ((((int)threadIdx.x) >> 3) * 5504)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 22016), ((((((((int)blockIdx.x) & 127) * 176128) + ((((int)threadIdx.x) >> 3) * 5504)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 22016), ((((((((int)blockIdx.x) & 127) * 176128) + ((((int)threadIdx.x) >> 3) * 5504)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 22016));
            int4 __61;
              int4 __62 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __63 = make_int4(2, 2, 2, 2);
              __61.x = (__62.x%__63.x);
              __61.y = (__62.y%__63.y);
              __61.z = (__62.z%__63.z);
              __61.w = (__62.w%__63.w);
            int4 __64;
              int4 __65 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __66 = make_int4(2, 2, 2, 2);
              __64.x = (__65.x/__66.x);
              __64.y = (__65.y/__66.y);
              __64.z = (__65.z/__66.z);
              __64.w = (__65.w/__66.w);
            int4 __67;
            ushort4 __68;
              ushort4 __69;
                ushort4 __70;
                  int4 __71 = make_int4(2, 2, 2, 2);
                  int4 __72 = make_int4(0, 0, 0, 0);
                  __70.x = (__71.x>=__72.x);
                  __70.y = (__71.y>=__72.y);
                  __70.z = (__71.z>=__72.z);
                  __70.w = (__71.w>=__72.w);
                ushort4 __73;
                  int4 __74 = make_int4(0, 0, 0, 0);
                  __73.x = (__61.x>=__74.x);
                  __73.y = (__61.y>=__74.y);
                  __73.z = (__61.z>=__74.z);
                  __73.w = (__61.w>=__74.w);
                __69.x = (__70.x&&__73.x);
                __69.y = (__70.y&&__73.y);
                __69.z = (__70.z&&__73.z);
                __69.w = (__70.w&&__73.w);
              ushort4 __75;
                ushort4 __76;
                  int4 __77 = make_int4(2, 2, 2, 2);
                  int4 __78 = make_int4(0, 0, 0, 0);
                  __76.x = (__77.x<__78.x);
                  __76.y = (__77.y<__78.y);
                  __76.z = (__77.z<__78.z);
                  __76.w = (__77.w<__78.w);
                ushort4 __79;
                  int4 __80 = make_int4(0, 0, 0, 0);
                  __79.x = (__61.x<=__80.x);
                  __79.y = (__61.y<=__80.y);
                  __79.z = (__61.z<=__80.z);
                  __79.w = (__61.w<=__80.w);
                __75.x = (__76.x&&__79.x);
                __75.y = (__76.y&&__79.y);
                __75.z = (__76.z&&__79.z);
                __75.w = (__76.w&&__79.w);
              __68.x = (__69.x||__75.x);
              __68.y = (__69.y||__75.y);
              __68.z = (__69.z||__75.z);
              __68.w = (__69.w||__75.w);
            int4 __81;
              int4 __82 = make_int4(1, 1, 1, 1);
              __81.x = (__64.x-__82.x);
              __81.y = (__64.y-__82.y);
              __81.z = (__64.z-__82.z);
              __81.w = (__64.w-__82.w);
            __67.x = (bool(__68.x)?__64.x:__81.x);
            __67.y = (bool(__68.y)?__64.y:__81.y);
            __67.z = (bool(__68.z)?__64.z:__81.z);
            __67.w = (bool(__68.w)?__64.w:__81.w);
            __59.x = (__60.x+__67.x);
            __59.y = (__60.y+__67.y);
            __59.z = (__60.z+__67.z);
            __59.w = (__60.w+__67.w);
          int __83 = ((0x000000ff << 0) & (B[__59.x] << 0))|((0x000000ff << 8) & (B[__59.y] << 8))|((0x000000ff << 16) & (B[__59.z] << 16))|((0x000000ff << 24) & (B[__59.w] << 24));
          int __84;
          int4 __85;
            int4 __86;
              int4 __87 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __88 = make_int4(2, 2, 2, 2);
              __86.x = (__87.x%__88.x);
              __86.y = (__87.y%__88.y);
              __86.z = (__87.z%__88.z);
              __86.w = (__87.w%__88.w);
            int4 __89;
            ushort4 __90;
              ushort4 __91;
                ushort4 __92;
                  int4 __93 = make_int4(2, 2, 2, 2);
                  int4 __94 = make_int4(0, 0, 0, 0);
                  __92.x = (__93.x>=__94.x);
                  __92.y = (__93.y>=__94.y);
                  __92.z = (__93.z>=__94.z);
                  __92.w = (__93.w>=__94.w);
                ushort4 __95;
                  int4 __96 = make_int4(0, 0, 0, 0);
                  __95.x = (__86.x>=__96.x);
                  __95.y = (__86.y>=__96.y);
                  __95.z = (__86.z>=__96.z);
                  __95.w = (__86.w>=__96.w);
                __91.x = (__92.x&&__95.x);
                __91.y = (__92.y&&__95.y);
                __91.z = (__92.z&&__95.z);
                __91.w = (__92.w&&__95.w);
              ushort4 __97;
                ushort4 __98;
                  int4 __99 = make_int4(2, 2, 2, 2);
                  int4 __100 = make_int4(0, 0, 0, 0);
                  __98.x = (__99.x<__100.x);
                  __98.y = (__99.y<__100.y);
                  __98.z = (__99.z<__100.z);
                  __98.w = (__99.w<__100.w);
                ushort4 __101;
                  int4 __102 = make_int4(0, 0, 0, 0);
                  __101.x = (__86.x<=__102.x);
                  __101.y = (__86.y<=__102.y);
                  __101.z = (__86.z<=__102.z);
                  __101.w = (__86.w<=__102.w);
                __97.x = (__98.x&&__101.x);
                __97.y = (__98.y&&__101.y);
                __97.z = (__98.z&&__101.z);
                __97.w = (__98.w&&__101.w);
              __90.x = (__91.x||__97.x);
              __90.y = (__91.y||__97.y);
              __90.z = (__91.z||__97.z);
              __90.w = (__91.w||__97.w);
            int4 __103;
              int4 __104 = make_int4(2, 2, 2, 2);
              __103.x = (__86.x+__104.x);
              __103.y = (__86.y+__104.y);
              __103.z = (__86.z+__104.z);
              __103.w = (__86.w+__104.w);
            __89.x = (bool(__90.x)?__86.x:__103.x);
            __89.y = (bool(__90.y)?__86.y:__103.y);
            __89.z = (bool(__90.z)?__86.z:__103.z);
            __89.w = (bool(__90.w)?__86.w:__103.w);
            int4 __105 = make_int4(4, 4, 4, 4);
            __85.x = (__89.x*__105.x);
            __85.y = (__89.y*__105.y);
            __85.z = (__89.z*__105.z);
            __85.w = (__89.w*__105.w);
          __84=((signed char)(__85.x) << 0);
          __84=__84 & ~(0x000000ff << 8) |((signed char)(__85.y) << 8);
          __84=__84 & ~(0x000000ff << 16) |((signed char)(__85.z) << 16);
          __84=__84 & ~(0x000000ff << 24) |((signed char)(__85.w) << 24);
          __58=((((char)(__83 >> 0)) >> ((char)(__84 >> 0))) << 0);
          __58=__58 & ~(0x000000ff << 8) |((((char)(__83 >> 8)) >> ((char)(__84 >> 8))) << 8);
          __58=__58 & ~(0x000000ff << 16) |((((char)(__83 >> 16)) >> ((char)(__84 >> 16))) << 16);
          __58=__58 & ~(0x000000ff << 24) |((((char)(__83 >> 24)) >> ((char)(__84 >> 24))) << 24);
        int __106 = (int)252645135;
        __57=((((char)(__58 >> 0)) & ((char)(__106 >> 0))) << 0);
        __57=__57 & ~(0x000000ff << 8) |((((char)(__58 >> 8)) & ((char)(__106 >> 8))) << 8);
        __57=__57 & ~(0x000000ff << 16) |((((char)(__58 >> 16)) & ((char)(__106 >> 16))) << 16);
        __57=__57 & ~(0x000000ff << 24) |((((char)(__58 >> 24)) & ((char)(__106 >> 24))) << 24);
      __56.x = (int)(((char)(__57 >> 0)));
      __56.y = (int)(((char)(__57 >> 8)));
      __56.z = (int)(((char)(__57 >> 16)));
      __56.w = (int)(((char)(__57 >> 24)));
      uint2 __107 = make_uint2(__pack_half2(LUT[__56.x],LUT[__56.y]),__pack_half2(LUT[__56.z],LUT[__56.w]));
      uint2 __108 = make_uint2(__pack_half2(Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 4)], Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 4)]), __pack_half2(Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 4)], Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 4)]));
      ((half2*)(&(__55.x)))->x = (((half2*)(&(__107.x)))->x*((half2*)(&(__108.x)))->x);
      ((half2*)(&(__55.x)))->y = (((half2*)(&(__107.x)))->y*((half2*)(&(__108.x)))->y);
      ((half2*)(&(__55.y)))->x = (((half2*)(&(__107.y)))->x*((half2*)(&(__108.y)))->x);
      ((half2*)(&(__55.y)))->y = (((half2*)(&(__107.y)))->y*((half2*)(&(__108.y)))->y);
    *(uint2*)(B_decode_shared + ((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 4)) + 160)) = __55;
    uint2 __109;
      int4 __110;
      int __111;
        int __112;
          int4 __113;
            int4 __114 = make_int4(((((((((int)blockIdx.x) & 127) * 176128) + ((((int)threadIdx.x) >> 3) * 5504)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 44032), ((((((((int)blockIdx.x) & 127) * 176128) + ((((int)threadIdx.x) >> 3) * 5504)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 44032), ((((((((int)blockIdx.x) & 127) * 176128) + ((((int)threadIdx.x) >> 3) * 5504)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 44032), ((((((((int)blockIdx.x) & 127) * 176128) + ((((int)threadIdx.x) >> 3) * 5504)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 44032));
            int4 __115;
              int4 __116 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __117 = make_int4(2, 2, 2, 2);
              __115.x = (__116.x%__117.x);
              __115.y = (__116.y%__117.y);
              __115.z = (__116.z%__117.z);
              __115.w = (__116.w%__117.w);
            int4 __118;
              int4 __119 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __120 = make_int4(2, 2, 2, 2);
              __118.x = (__119.x/__120.x);
              __118.y = (__119.y/__120.y);
              __118.z = (__119.z/__120.z);
              __118.w = (__119.w/__120.w);
            int4 __121;
            ushort4 __122;
              ushort4 __123;
                ushort4 __124;
                  int4 __125 = make_int4(2, 2, 2, 2);
                  int4 __126 = make_int4(0, 0, 0, 0);
                  __124.x = (__125.x>=__126.x);
                  __124.y = (__125.y>=__126.y);
                  __124.z = (__125.z>=__126.z);
                  __124.w = (__125.w>=__126.w);
                ushort4 __127;
                  int4 __128 = make_int4(0, 0, 0, 0);
                  __127.x = (__115.x>=__128.x);
                  __127.y = (__115.y>=__128.y);
                  __127.z = (__115.z>=__128.z);
                  __127.w = (__115.w>=__128.w);
                __123.x = (__124.x&&__127.x);
                __123.y = (__124.y&&__127.y);
                __123.z = (__124.z&&__127.z);
                __123.w = (__124.w&&__127.w);
              ushort4 __129;
                ushort4 __130;
                  int4 __131 = make_int4(2, 2, 2, 2);
                  int4 __132 = make_int4(0, 0, 0, 0);
                  __130.x = (__131.x<__132.x);
                  __130.y = (__131.y<__132.y);
                  __130.z = (__131.z<__132.z);
                  __130.w = (__131.w<__132.w);
                ushort4 __133;
                  int4 __134 = make_int4(0, 0, 0, 0);
                  __133.x = (__115.x<=__134.x);
                  __133.y = (__115.y<=__134.y);
                  __133.z = (__115.z<=__134.z);
                  __133.w = (__115.w<=__134.w);
                __129.x = (__130.x&&__133.x);
                __129.y = (__130.y&&__133.y);
                __129.z = (__130.z&&__133.z);
                __129.w = (__130.w&&__133.w);
              __122.x = (__123.x||__129.x);
              __122.y = (__123.y||__129.y);
              __122.z = (__123.z||__129.z);
              __122.w = (__123.w||__129.w);
            int4 __135;
              int4 __136 = make_int4(1, 1, 1, 1);
              __135.x = (__118.x-__136.x);
              __135.y = (__118.y-__136.y);
              __135.z = (__118.z-__136.z);
              __135.w = (__118.w-__136.w);
            __121.x = (bool(__122.x)?__118.x:__135.x);
            __121.y = (bool(__122.y)?__118.y:__135.y);
            __121.z = (bool(__122.z)?__118.z:__135.z);
            __121.w = (bool(__122.w)?__118.w:__135.w);
            __113.x = (__114.x+__121.x);
            __113.y = (__114.y+__121.y);
            __113.z = (__114.z+__121.z);
            __113.w = (__114.w+__121.w);
          int __137 = ((0x000000ff << 0) & (B[__113.x] << 0))|((0x000000ff << 8) & (B[__113.y] << 8))|((0x000000ff << 16) & (B[__113.z] << 16))|((0x000000ff << 24) & (B[__113.w] << 24));
          int __138;
          int4 __139;
            int4 __140;
              int4 __141 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __142 = make_int4(2, 2, 2, 2);
              __140.x = (__141.x%__142.x);
              __140.y = (__141.y%__142.y);
              __140.z = (__141.z%__142.z);
              __140.w = (__141.w%__142.w);
            int4 __143;
            ushort4 __144;
              ushort4 __145;
                ushort4 __146;
                  int4 __147 = make_int4(2, 2, 2, 2);
                  int4 __148 = make_int4(0, 0, 0, 0);
                  __146.x = (__147.x>=__148.x);
                  __146.y = (__147.y>=__148.y);
                  __146.z = (__147.z>=__148.z);
                  __146.w = (__147.w>=__148.w);
                ushort4 __149;
                  int4 __150 = make_int4(0, 0, 0, 0);
                  __149.x = (__140.x>=__150.x);
                  __149.y = (__140.y>=__150.y);
                  __149.z = (__140.z>=__150.z);
                  __149.w = (__140.w>=__150.w);
                __145.x = (__146.x&&__149.x);
                __145.y = (__146.y&&__149.y);
                __145.z = (__146.z&&__149.z);
                __145.w = (__146.w&&__149.w);
              ushort4 __151;
                ushort4 __152;
                  int4 __153 = make_int4(2, 2, 2, 2);
                  int4 __154 = make_int4(0, 0, 0, 0);
                  __152.x = (__153.x<__154.x);
                  __152.y = (__153.y<__154.y);
                  __152.z = (__153.z<__154.z);
                  __152.w = (__153.w<__154.w);
                ushort4 __155;
                  int4 __156 = make_int4(0, 0, 0, 0);
                  __155.x = (__140.x<=__156.x);
                  __155.y = (__140.y<=__156.y);
                  __155.z = (__140.z<=__156.z);
                  __155.w = (__140.w<=__156.w);
                __151.x = (__152.x&&__155.x);
                __151.y = (__152.y&&__155.y);
                __151.z = (__152.z&&__155.z);
                __151.w = (__152.w&&__155.w);
              __144.x = (__145.x||__151.x);
              __144.y = (__145.y||__151.y);
              __144.z = (__145.z||__151.z);
              __144.w = (__145.w||__151.w);
            int4 __157;
              int4 __158 = make_int4(2, 2, 2, 2);
              __157.x = (__140.x+__158.x);
              __157.y = (__140.y+__158.y);
              __157.z = (__140.z+__158.z);
              __157.w = (__140.w+__158.w);
            __143.x = (bool(__144.x)?__140.x:__157.x);
            __143.y = (bool(__144.y)?__140.y:__157.y);
            __143.z = (bool(__144.z)?__140.z:__157.z);
            __143.w = (bool(__144.w)?__140.w:__157.w);
            int4 __159 = make_int4(4, 4, 4, 4);
            __139.x = (__143.x*__159.x);
            __139.y = (__143.y*__159.y);
            __139.z = (__143.z*__159.z);
            __139.w = (__143.w*__159.w);
          __138=((signed char)(__139.x) << 0);
          __138=__138 & ~(0x000000ff << 8) |((signed char)(__139.y) << 8);
          __138=__138 & ~(0x000000ff << 16) |((signed char)(__139.z) << 16);
          __138=__138 & ~(0x000000ff << 24) |((signed char)(__139.w) << 24);
          __112=((((char)(__137 >> 0)) >> ((char)(__138 >> 0))) << 0);
          __112=__112 & ~(0x000000ff << 8) |((((char)(__137 >> 8)) >> ((char)(__138 >> 8))) << 8);
          __112=__112 & ~(0x000000ff << 16) |((((char)(__137 >> 16)) >> ((char)(__138 >> 16))) << 16);
          __112=__112 & ~(0x000000ff << 24) |((((char)(__137 >> 24)) >> ((char)(__138 >> 24))) << 24);
        int __160 = (int)252645135;
        __111=((((char)(__112 >> 0)) & ((char)(__160 >> 0))) << 0);
        __111=__111 & ~(0x000000ff << 8) |((((char)(__112 >> 8)) & ((char)(__160 >> 8))) << 8);
        __111=__111 & ~(0x000000ff << 16) |((((char)(__112 >> 16)) & ((char)(__160 >> 16))) << 16);
        __111=__111 & ~(0x000000ff << 24) |((((char)(__112 >> 24)) & ((char)(__160 >> 24))) << 24);
      __110.x = (int)(((char)(__111 >> 0)));
      __110.y = (int)(((char)(__111 >> 8)));
      __110.z = (int)(((char)(__111 >> 16)));
      __110.w = (int)(((char)(__111 >> 24)));
      uint2 __161 = make_uint2(__pack_half2(LUT[__110.x],LUT[__110.y]),__pack_half2(LUT[__110.z],LUT[__110.w]));
      uint2 __162 = make_uint2(__pack_half2(Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 8)], Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 8)]), __pack_half2(Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 8)], Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 8)]));
      ((half2*)(&(__109.x)))->x = (((half2*)(&(__161.x)))->x*((half2*)(&(__162.x)))->x);
      ((half2*)(&(__109.x)))->y = (((half2*)(&(__161.x)))->y*((half2*)(&(__162.x)))->y);
      ((half2*)(&(__109.y)))->x = (((half2*)(&(__161.y)))->x*((half2*)(&(__162.y)))->x);
      ((half2*)(&(__109.y)))->y = (((half2*)(&(__161.y)))->y*((half2*)(&(__162.y)))->y);
    *(uint2*)(B_decode_shared + ((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 4)) + 320)) = __109;
    uint2 __163;
      int4 __164;
      int __165;
        int __166;
          int4 __167;
            int4 __168 = make_int4(((((((((int)blockIdx.x) & 127) * 176128) + ((((int)threadIdx.x) >> 3) * 5504)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 66048), ((((((((int)blockIdx.x) & 127) * 176128) + ((((int)threadIdx.x) >> 3) * 5504)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 66048), ((((((((int)blockIdx.x) & 127) * 176128) + ((((int)threadIdx.x) >> 3) * 5504)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 66048), ((((((((int)blockIdx.x) & 127) * 176128) + ((((int)threadIdx.x) >> 3) * 5504)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 66048));
            int4 __169;
              int4 __170 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __171 = make_int4(2, 2, 2, 2);
              __169.x = (__170.x%__171.x);
              __169.y = (__170.y%__171.y);
              __169.z = (__170.z%__171.z);
              __169.w = (__170.w%__171.w);
            int4 __172;
              int4 __173 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __174 = make_int4(2, 2, 2, 2);
              __172.x = (__173.x/__174.x);
              __172.y = (__173.y/__174.y);
              __172.z = (__173.z/__174.z);
              __172.w = (__173.w/__174.w);
            int4 __175;
            ushort4 __176;
              ushort4 __177;
                ushort4 __178;
                  int4 __179 = make_int4(2, 2, 2, 2);
                  int4 __180 = make_int4(0, 0, 0, 0);
                  __178.x = (__179.x>=__180.x);
                  __178.y = (__179.y>=__180.y);
                  __178.z = (__179.z>=__180.z);
                  __178.w = (__179.w>=__180.w);
                ushort4 __181;
                  int4 __182 = make_int4(0, 0, 0, 0);
                  __181.x = (__169.x>=__182.x);
                  __181.y = (__169.y>=__182.y);
                  __181.z = (__169.z>=__182.z);
                  __181.w = (__169.w>=__182.w);
                __177.x = (__178.x&&__181.x);
                __177.y = (__178.y&&__181.y);
                __177.z = (__178.z&&__181.z);
                __177.w = (__178.w&&__181.w);
              ushort4 __183;
                ushort4 __184;
                  int4 __185 = make_int4(2, 2, 2, 2);
                  int4 __186 = make_int4(0, 0, 0, 0);
                  __184.x = (__185.x<__186.x);
                  __184.y = (__185.y<__186.y);
                  __184.z = (__185.z<__186.z);
                  __184.w = (__185.w<__186.w);
                ushort4 __187;
                  int4 __188 = make_int4(0, 0, 0, 0);
                  __187.x = (__169.x<=__188.x);
                  __187.y = (__169.y<=__188.y);
                  __187.z = (__169.z<=__188.z);
                  __187.w = (__169.w<=__188.w);
                __183.x = (__184.x&&__187.x);
                __183.y = (__184.y&&__187.y);
                __183.z = (__184.z&&__187.z);
                __183.w = (__184.w&&__187.w);
              __176.x = (__177.x||__183.x);
              __176.y = (__177.y||__183.y);
              __176.z = (__177.z||__183.z);
              __176.w = (__177.w||__183.w);
            int4 __189;
              int4 __190 = make_int4(1, 1, 1, 1);
              __189.x = (__172.x-__190.x);
              __189.y = (__172.y-__190.y);
              __189.z = (__172.z-__190.z);
              __189.w = (__172.w-__190.w);
            __175.x = (bool(__176.x)?__172.x:__189.x);
            __175.y = (bool(__176.y)?__172.y:__189.y);
            __175.z = (bool(__176.z)?__172.z:__189.z);
            __175.w = (bool(__176.w)?__172.w:__189.w);
            __167.x = (__168.x+__175.x);
            __167.y = (__168.y+__175.y);
            __167.z = (__168.z+__175.z);
            __167.w = (__168.w+__175.w);
          int __191 = ((0x000000ff << 0) & (B[__167.x] << 0))|((0x000000ff << 8) & (B[__167.y] << 8))|((0x000000ff << 16) & (B[__167.z] << 16))|((0x000000ff << 24) & (B[__167.w] << 24));
          int __192;
          int4 __193;
            int4 __194;
              int4 __195 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __196 = make_int4(2, 2, 2, 2);
              __194.x = (__195.x%__196.x);
              __194.y = (__195.y%__196.y);
              __194.z = (__195.z%__196.z);
              __194.w = (__195.w%__196.w);
            int4 __197;
            ushort4 __198;
              ushort4 __199;
                ushort4 __200;
                  int4 __201 = make_int4(2, 2, 2, 2);
                  int4 __202 = make_int4(0, 0, 0, 0);
                  __200.x = (__201.x>=__202.x);
                  __200.y = (__201.y>=__202.y);
                  __200.z = (__201.z>=__202.z);
                  __200.w = (__201.w>=__202.w);
                ushort4 __203;
                  int4 __204 = make_int4(0, 0, 0, 0);
                  __203.x = (__194.x>=__204.x);
                  __203.y = (__194.y>=__204.y);
                  __203.z = (__194.z>=__204.z);
                  __203.w = (__194.w>=__204.w);
                __199.x = (__200.x&&__203.x);
                __199.y = (__200.y&&__203.y);
                __199.z = (__200.z&&__203.z);
                __199.w = (__200.w&&__203.w);
              ushort4 __205;
                ushort4 __206;
                  int4 __207 = make_int4(2, 2, 2, 2);
                  int4 __208 = make_int4(0, 0, 0, 0);
                  __206.x = (__207.x<__208.x);
                  __206.y = (__207.y<__208.y);
                  __206.z = (__207.z<__208.z);
                  __206.w = (__207.w<__208.w);
                ushort4 __209;
                  int4 __210 = make_int4(0, 0, 0, 0);
                  __209.x = (__194.x<=__210.x);
                  __209.y = (__194.y<=__210.y);
                  __209.z = (__194.z<=__210.z);
                  __209.w = (__194.w<=__210.w);
                __205.x = (__206.x&&__209.x);
                __205.y = (__206.y&&__209.y);
                __205.z = (__206.z&&__209.z);
                __205.w = (__206.w&&__209.w);
              __198.x = (__199.x||__205.x);
              __198.y = (__199.y||__205.y);
              __198.z = (__199.z||__205.z);
              __198.w = (__199.w||__205.w);
            int4 __211;
              int4 __212 = make_int4(2, 2, 2, 2);
              __211.x = (__194.x+__212.x);
              __211.y = (__194.y+__212.y);
              __211.z = (__194.z+__212.z);
              __211.w = (__194.w+__212.w);
            __197.x = (bool(__198.x)?__194.x:__211.x);
            __197.y = (bool(__198.y)?__194.y:__211.y);
            __197.z = (bool(__198.z)?__194.z:__211.z);
            __197.w = (bool(__198.w)?__194.w:__211.w);
            int4 __213 = make_int4(4, 4, 4, 4);
            __193.x = (__197.x*__213.x);
            __193.y = (__197.y*__213.y);
            __193.z = (__197.z*__213.z);
            __193.w = (__197.w*__213.w);
          __192=((signed char)(__193.x) << 0);
          __192=__192 & ~(0x000000ff << 8) |((signed char)(__193.y) << 8);
          __192=__192 & ~(0x000000ff << 16) |((signed char)(__193.z) << 16);
          __192=__192 & ~(0x000000ff << 24) |((signed char)(__193.w) << 24);
          __166=((((char)(__191 >> 0)) >> ((char)(__192 >> 0))) << 0);
          __166=__166 & ~(0x000000ff << 8) |((((char)(__191 >> 8)) >> ((char)(__192 >> 8))) << 8);
          __166=__166 & ~(0x000000ff << 16) |((((char)(__191 >> 16)) >> ((char)(__192 >> 16))) << 16);
          __166=__166 & ~(0x000000ff << 24) |((((char)(__191 >> 24)) >> ((char)(__192 >> 24))) << 24);
        int __214 = (int)252645135;
        __165=((((char)(__166 >> 0)) & ((char)(__214 >> 0))) << 0);
        __165=__165 & ~(0x000000ff << 8) |((((char)(__166 >> 8)) & ((char)(__214 >> 8))) << 8);
        __165=__165 & ~(0x000000ff << 16) |((((char)(__166 >> 16)) & ((char)(__214 >> 16))) << 16);
        __165=__165 & ~(0x000000ff << 24) |((((char)(__166 >> 24)) & ((char)(__214 >> 24))) << 24);
      __164.x = (int)(((char)(__165 >> 0)));
      __164.y = (int)(((char)(__165 >> 8)));
      __164.z = (int)(((char)(__165 >> 16)));
      __164.w = (int)(((char)(__165 >> 24)));
      uint2 __215 = make_uint2(__pack_half2(LUT[__164.x],LUT[__164.y]),__pack_half2(LUT[__164.z],LUT[__164.w]));
      uint2 __216 = make_uint2(__pack_half2(Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 12)], Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 12)]), __pack_half2(Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 12)], Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 12)]));
      ((half2*)(&(__163.x)))->x = (((half2*)(&(__215.x)))->x*((half2*)(&(__216.x)))->x);
      ((half2*)(&(__163.x)))->y = (((half2*)(&(__215.x)))->y*((half2*)(&(__216.x)))->y);
      ((half2*)(&(__163.y)))->x = (((half2*)(&(__215.y)))->x*((half2*)(&(__216.y)))->x);
      ((half2*)(&(__163.y)))->y = (((half2*)(&(__215.y)))->y*((half2*)(&(__216.y)))->y);
    *(uint2*)(B_decode_shared + ((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 4)) + 480)) = __163;
    uint2 __217;
      int4 __218;
      int __219;
        int __220;
          int4 __221;
            int4 __222 = make_int4(((((((((int)blockIdx.x) & 127) * 176128) + ((((int)threadIdx.x) >> 3) * 5504)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 88064), ((((((((int)blockIdx.x) & 127) * 176128) + ((((int)threadIdx.x) >> 3) * 5504)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 88064), ((((((((int)blockIdx.x) & 127) * 176128) + ((((int)threadIdx.x) >> 3) * 5504)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 88064), ((((((((int)blockIdx.x) & 127) * 176128) + ((((int)threadIdx.x) >> 3) * 5504)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 88064));
            int4 __223;
              int4 __224 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __225 = make_int4(2, 2, 2, 2);
              __223.x = (__224.x%__225.x);
              __223.y = (__224.y%__225.y);
              __223.z = (__224.z%__225.z);
              __223.w = (__224.w%__225.w);
            int4 __226;
              int4 __227 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __228 = make_int4(2, 2, 2, 2);
              __226.x = (__227.x/__228.x);
              __226.y = (__227.y/__228.y);
              __226.z = (__227.z/__228.z);
              __226.w = (__227.w/__228.w);
            int4 __229;
            ushort4 __230;
              ushort4 __231;
                ushort4 __232;
                  int4 __233 = make_int4(2, 2, 2, 2);
                  int4 __234 = make_int4(0, 0, 0, 0);
                  __232.x = (__233.x>=__234.x);
                  __232.y = (__233.y>=__234.y);
                  __232.z = (__233.z>=__234.z);
                  __232.w = (__233.w>=__234.w);
                ushort4 __235;
                  int4 __236 = make_int4(0, 0, 0, 0);
                  __235.x = (__223.x>=__236.x);
                  __235.y = (__223.y>=__236.y);
                  __235.z = (__223.z>=__236.z);
                  __235.w = (__223.w>=__236.w);
                __231.x = (__232.x&&__235.x);
                __231.y = (__232.y&&__235.y);
                __231.z = (__232.z&&__235.z);
                __231.w = (__232.w&&__235.w);
              ushort4 __237;
                ushort4 __238;
                  int4 __239 = make_int4(2, 2, 2, 2);
                  int4 __240 = make_int4(0, 0, 0, 0);
                  __238.x = (__239.x<__240.x);
                  __238.y = (__239.y<__240.y);
                  __238.z = (__239.z<__240.z);
                  __238.w = (__239.w<__240.w);
                ushort4 __241;
                  int4 __242 = make_int4(0, 0, 0, 0);
                  __241.x = (__223.x<=__242.x);
                  __241.y = (__223.y<=__242.y);
                  __241.z = (__223.z<=__242.z);
                  __241.w = (__223.w<=__242.w);
                __237.x = (__238.x&&__241.x);
                __237.y = (__238.y&&__241.y);
                __237.z = (__238.z&&__241.z);
                __237.w = (__238.w&&__241.w);
              __230.x = (__231.x||__237.x);
              __230.y = (__231.y||__237.y);
              __230.z = (__231.z||__237.z);
              __230.w = (__231.w||__237.w);
            int4 __243;
              int4 __244 = make_int4(1, 1, 1, 1);
              __243.x = (__226.x-__244.x);
              __243.y = (__226.y-__244.y);
              __243.z = (__226.z-__244.z);
              __243.w = (__226.w-__244.w);
            __229.x = (bool(__230.x)?__226.x:__243.x);
            __229.y = (bool(__230.y)?__226.y:__243.y);
            __229.z = (bool(__230.z)?__226.z:__243.z);
            __229.w = (bool(__230.w)?__226.w:__243.w);
            __221.x = (__222.x+__229.x);
            __221.y = (__222.y+__229.y);
            __221.z = (__222.z+__229.z);
            __221.w = (__222.w+__229.w);
          int __245 = ((0x000000ff << 0) & (B[__221.x] << 0))|((0x000000ff << 8) & (B[__221.y] << 8))|((0x000000ff << 16) & (B[__221.z] << 16))|((0x000000ff << 24) & (B[__221.w] << 24));
          int __246;
          int4 __247;
            int4 __248;
              int4 __249 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __250 = make_int4(2, 2, 2, 2);
              __248.x = (__249.x%__250.x);
              __248.y = (__249.y%__250.y);
              __248.z = (__249.z%__250.z);
              __248.w = (__249.w%__250.w);
            int4 __251;
            ushort4 __252;
              ushort4 __253;
                ushort4 __254;
                  int4 __255 = make_int4(2, 2, 2, 2);
                  int4 __256 = make_int4(0, 0, 0, 0);
                  __254.x = (__255.x>=__256.x);
                  __254.y = (__255.y>=__256.y);
                  __254.z = (__255.z>=__256.z);
                  __254.w = (__255.w>=__256.w);
                ushort4 __257;
                  int4 __258 = make_int4(0, 0, 0, 0);
                  __257.x = (__248.x>=__258.x);
                  __257.y = (__248.y>=__258.y);
                  __257.z = (__248.z>=__258.z);
                  __257.w = (__248.w>=__258.w);
                __253.x = (__254.x&&__257.x);
                __253.y = (__254.y&&__257.y);
                __253.z = (__254.z&&__257.z);
                __253.w = (__254.w&&__257.w);
              ushort4 __259;
                ushort4 __260;
                  int4 __261 = make_int4(2, 2, 2, 2);
                  int4 __262 = make_int4(0, 0, 0, 0);
                  __260.x = (__261.x<__262.x);
                  __260.y = (__261.y<__262.y);
                  __260.z = (__261.z<__262.z);
                  __260.w = (__261.w<__262.w);
                ushort4 __263;
                  int4 __264 = make_int4(0, 0, 0, 0);
                  __263.x = (__248.x<=__264.x);
                  __263.y = (__248.y<=__264.y);
                  __263.z = (__248.z<=__264.z);
                  __263.w = (__248.w<=__264.w);
                __259.x = (__260.x&&__263.x);
                __259.y = (__260.y&&__263.y);
                __259.z = (__260.z&&__263.z);
                __259.w = (__260.w&&__263.w);
              __252.x = (__253.x||__259.x);
              __252.y = (__253.y||__259.y);
              __252.z = (__253.z||__259.z);
              __252.w = (__253.w||__259.w);
            int4 __265;
              int4 __266 = make_int4(2, 2, 2, 2);
              __265.x = (__248.x+__266.x);
              __265.y = (__248.y+__266.y);
              __265.z = (__248.z+__266.z);
              __265.w = (__248.w+__266.w);
            __251.x = (bool(__252.x)?__248.x:__265.x);
            __251.y = (bool(__252.y)?__248.y:__265.y);
            __251.z = (bool(__252.z)?__248.z:__265.z);
            __251.w = (bool(__252.w)?__248.w:__265.w);
            int4 __267 = make_int4(4, 4, 4, 4);
            __247.x = (__251.x*__267.x);
            __247.y = (__251.y*__267.y);
            __247.z = (__251.z*__267.z);
            __247.w = (__251.w*__267.w);
          __246=((signed char)(__247.x) << 0);
          __246=__246 & ~(0x000000ff << 8) |((signed char)(__247.y) << 8);
          __246=__246 & ~(0x000000ff << 16) |((signed char)(__247.z) << 16);
          __246=__246 & ~(0x000000ff << 24) |((signed char)(__247.w) << 24);
          __220=((((char)(__245 >> 0)) >> ((char)(__246 >> 0))) << 0);
          __220=__220 & ~(0x000000ff << 8) |((((char)(__245 >> 8)) >> ((char)(__246 >> 8))) << 8);
          __220=__220 & ~(0x000000ff << 16) |((((char)(__245 >> 16)) >> ((char)(__246 >> 16))) << 16);
          __220=__220 & ~(0x000000ff << 24) |((((char)(__245 >> 24)) >> ((char)(__246 >> 24))) << 24);
        int __268 = (int)252645135;
        __219=((((char)(__220 >> 0)) & ((char)(__268 >> 0))) << 0);
        __219=__219 & ~(0x000000ff << 8) |((((char)(__220 >> 8)) & ((char)(__268 >> 8))) << 8);
        __219=__219 & ~(0x000000ff << 16) |((((char)(__220 >> 16)) & ((char)(__268 >> 16))) << 16);
        __219=__219 & ~(0x000000ff << 24) |((((char)(__220 >> 24)) & ((char)(__268 >> 24))) << 24);
      __218.x = (int)(((char)(__219 >> 0)));
      __218.y = (int)(((char)(__219 >> 8)));
      __218.z = (int)(((char)(__219 >> 16)));
      __218.w = (int)(((char)(__219 >> 24)));
      uint2 __269 = make_uint2(__pack_half2(LUT[__218.x],LUT[__218.y]),__pack_half2(LUT[__218.z],LUT[__218.w]));
      uint2 __270 = make_uint2(__pack_half2(Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 16)], Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 16)]), __pack_half2(Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 16)], Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 16)]));
      ((half2*)(&(__217.x)))->x = (((half2*)(&(__269.x)))->x*((half2*)(&(__270.x)))->x);
      ((half2*)(&(__217.x)))->y = (((half2*)(&(__269.x)))->y*((half2*)(&(__270.x)))->y);
      ((half2*)(&(__217.y)))->x = (((half2*)(&(__269.y)))->x*((half2*)(&(__270.y)))->x);
      ((half2*)(&(__217.y)))->y = (((half2*)(&(__269.y)))->y*((half2*)(&(__270.y)))->y);
    *(uint2*)(B_decode_shared + ((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 4)) + 640)) = __217;
    uint2 __271;
      int4 __272;
      int __273;
        int __274;
          int4 __275;
            int4 __276 = make_int4(((((((((int)blockIdx.x) & 127) * 176128) + ((((int)threadIdx.x) >> 3) * 5504)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 110080), ((((((((int)blockIdx.x) & 127) * 176128) + ((((int)threadIdx.x) >> 3) * 5504)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 110080), ((((((((int)blockIdx.x) & 127) * 176128) + ((((int)threadIdx.x) >> 3) * 5504)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 110080), ((((((((int)blockIdx.x) & 127) * 176128) + ((((int)threadIdx.x) >> 3) * 5504)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 110080));
            int4 __277;
              int4 __278 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __279 = make_int4(2, 2, 2, 2);
              __277.x = (__278.x%__279.x);
              __277.y = (__278.y%__279.y);
              __277.z = (__278.z%__279.z);
              __277.w = (__278.w%__279.w);
            int4 __280;
              int4 __281 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __282 = make_int4(2, 2, 2, 2);
              __280.x = (__281.x/__282.x);
              __280.y = (__281.y/__282.y);
              __280.z = (__281.z/__282.z);
              __280.w = (__281.w/__282.w);
            int4 __283;
            ushort4 __284;
              ushort4 __285;
                ushort4 __286;
                  int4 __287 = make_int4(2, 2, 2, 2);
                  int4 __288 = make_int4(0, 0, 0, 0);
                  __286.x = (__287.x>=__288.x);
                  __286.y = (__287.y>=__288.y);
                  __286.z = (__287.z>=__288.z);
                  __286.w = (__287.w>=__288.w);
                ushort4 __289;
                  int4 __290 = make_int4(0, 0, 0, 0);
                  __289.x = (__277.x>=__290.x);
                  __289.y = (__277.y>=__290.y);
                  __289.z = (__277.z>=__290.z);
                  __289.w = (__277.w>=__290.w);
                __285.x = (__286.x&&__289.x);
                __285.y = (__286.y&&__289.y);
                __285.z = (__286.z&&__289.z);
                __285.w = (__286.w&&__289.w);
              ushort4 __291;
                ushort4 __292;
                  int4 __293 = make_int4(2, 2, 2, 2);
                  int4 __294 = make_int4(0, 0, 0, 0);
                  __292.x = (__293.x<__294.x);
                  __292.y = (__293.y<__294.y);
                  __292.z = (__293.z<__294.z);
                  __292.w = (__293.w<__294.w);
                ushort4 __295;
                  int4 __296 = make_int4(0, 0, 0, 0);
                  __295.x = (__277.x<=__296.x);
                  __295.y = (__277.y<=__296.y);
                  __295.z = (__277.z<=__296.z);
                  __295.w = (__277.w<=__296.w);
                __291.x = (__292.x&&__295.x);
                __291.y = (__292.y&&__295.y);
                __291.z = (__292.z&&__295.z);
                __291.w = (__292.w&&__295.w);
              __284.x = (__285.x||__291.x);
              __284.y = (__285.y||__291.y);
              __284.z = (__285.z||__291.z);
              __284.w = (__285.w||__291.w);
            int4 __297;
              int4 __298 = make_int4(1, 1, 1, 1);
              __297.x = (__280.x-__298.x);
              __297.y = (__280.y-__298.y);
              __297.z = (__280.z-__298.z);
              __297.w = (__280.w-__298.w);
            __283.x = (bool(__284.x)?__280.x:__297.x);
            __283.y = (bool(__284.y)?__280.y:__297.y);
            __283.z = (bool(__284.z)?__280.z:__297.z);
            __283.w = (bool(__284.w)?__280.w:__297.w);
            __275.x = (__276.x+__283.x);
            __275.y = (__276.y+__283.y);
            __275.z = (__276.z+__283.z);
            __275.w = (__276.w+__283.w);
          int __299 = ((0x000000ff << 0) & (B[__275.x] << 0))|((0x000000ff << 8) & (B[__275.y] << 8))|((0x000000ff << 16) & (B[__275.z] << 16))|((0x000000ff << 24) & (B[__275.w] << 24));
          int __300;
          int4 __301;
            int4 __302;
              int4 __303 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __304 = make_int4(2, 2, 2, 2);
              __302.x = (__303.x%__304.x);
              __302.y = (__303.y%__304.y);
              __302.z = (__303.z%__304.z);
              __302.w = (__303.w%__304.w);
            int4 __305;
            ushort4 __306;
              ushort4 __307;
                ushort4 __308;
                  int4 __309 = make_int4(2, 2, 2, 2);
                  int4 __310 = make_int4(0, 0, 0, 0);
                  __308.x = (__309.x>=__310.x);
                  __308.y = (__309.y>=__310.y);
                  __308.z = (__309.z>=__310.z);
                  __308.w = (__309.w>=__310.w);
                ushort4 __311;
                  int4 __312 = make_int4(0, 0, 0, 0);
                  __311.x = (__302.x>=__312.x);
                  __311.y = (__302.y>=__312.y);
                  __311.z = (__302.z>=__312.z);
                  __311.w = (__302.w>=__312.w);
                __307.x = (__308.x&&__311.x);
                __307.y = (__308.y&&__311.y);
                __307.z = (__308.z&&__311.z);
                __307.w = (__308.w&&__311.w);
              ushort4 __313;
                ushort4 __314;
                  int4 __315 = make_int4(2, 2, 2, 2);
                  int4 __316 = make_int4(0, 0, 0, 0);
                  __314.x = (__315.x<__316.x);
                  __314.y = (__315.y<__316.y);
                  __314.z = (__315.z<__316.z);
                  __314.w = (__315.w<__316.w);
                ushort4 __317;
                  int4 __318 = make_int4(0, 0, 0, 0);
                  __317.x = (__302.x<=__318.x);
                  __317.y = (__302.y<=__318.y);
                  __317.z = (__302.z<=__318.z);
                  __317.w = (__302.w<=__318.w);
                __313.x = (__314.x&&__317.x);
                __313.y = (__314.y&&__317.y);
                __313.z = (__314.z&&__317.z);
                __313.w = (__314.w&&__317.w);
              __306.x = (__307.x||__313.x);
              __306.y = (__307.y||__313.y);
              __306.z = (__307.z||__313.z);
              __306.w = (__307.w||__313.w);
            int4 __319;
              int4 __320 = make_int4(2, 2, 2, 2);
              __319.x = (__302.x+__320.x);
              __319.y = (__302.y+__320.y);
              __319.z = (__302.z+__320.z);
              __319.w = (__302.w+__320.w);
            __305.x = (bool(__306.x)?__302.x:__319.x);
            __305.y = (bool(__306.y)?__302.y:__319.y);
            __305.z = (bool(__306.z)?__302.z:__319.z);
            __305.w = (bool(__306.w)?__302.w:__319.w);
            int4 __321 = make_int4(4, 4, 4, 4);
            __301.x = (__305.x*__321.x);
            __301.y = (__305.y*__321.y);
            __301.z = (__305.z*__321.z);
            __301.w = (__305.w*__321.w);
          __300=((signed char)(__301.x) << 0);
          __300=__300 & ~(0x000000ff << 8) |((signed char)(__301.y) << 8);
          __300=__300 & ~(0x000000ff << 16) |((signed char)(__301.z) << 16);
          __300=__300 & ~(0x000000ff << 24) |((signed char)(__301.w) << 24);
          __274=((((char)(__299 >> 0)) >> ((char)(__300 >> 0))) << 0);
          __274=__274 & ~(0x000000ff << 8) |((((char)(__299 >> 8)) >> ((char)(__300 >> 8))) << 8);
          __274=__274 & ~(0x000000ff << 16) |((((char)(__299 >> 16)) >> ((char)(__300 >> 16))) << 16);
          __274=__274 & ~(0x000000ff << 24) |((((char)(__299 >> 24)) >> ((char)(__300 >> 24))) << 24);
        int __322 = (int)252645135;
        __273=((((char)(__274 >> 0)) & ((char)(__322 >> 0))) << 0);
        __273=__273 & ~(0x000000ff << 8) |((((char)(__274 >> 8)) & ((char)(__322 >> 8))) << 8);
        __273=__273 & ~(0x000000ff << 16) |((((char)(__274 >> 16)) & ((char)(__322 >> 16))) << 16);
        __273=__273 & ~(0x000000ff << 24) |((((char)(__274 >> 24)) & ((char)(__322 >> 24))) << 24);
      __272.x = (int)(((char)(__273 >> 0)));
      __272.y = (int)(((char)(__273 >> 8)));
      __272.z = (int)(((char)(__273 >> 16)));
      __272.w = (int)(((char)(__273 >> 24)));
      uint2 __323 = make_uint2(__pack_half2(LUT[__272.x],LUT[__272.y]),__pack_half2(LUT[__272.z],LUT[__272.w]));
      uint2 __324 = make_uint2(__pack_half2(Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 20)], Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 20)]), __pack_half2(Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 20)], Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 20)]));
      ((half2*)(&(__271.x)))->x = (((half2*)(&(__323.x)))->x*((half2*)(&(__324.x)))->x);
      ((half2*)(&(__271.x)))->y = (((half2*)(&(__323.x)))->y*((half2*)(&(__324.x)))->y);
      ((half2*)(&(__271.y)))->x = (((half2*)(&(__323.y)))->x*((half2*)(&(__324.y)))->x);
      ((half2*)(&(__271.y)))->y = (((half2*)(&(__323.y)))->y*((half2*)(&(__324.y)))->y);
    *(uint2*)(B_decode_shared + ((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 4)) + 800)) = __271;
    uint2 __325;
      int4 __326;
      int __327;
        int __328;
          int4 __329;
            int4 __330 = make_int4(((((((((int)blockIdx.x) & 127) * 176128) + ((((int)threadIdx.x) >> 3) * 5504)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 132096), ((((((((int)blockIdx.x) & 127) * 176128) + ((((int)threadIdx.x) >> 3) * 5504)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 132096), ((((((((int)blockIdx.x) & 127) * 176128) + ((((int)threadIdx.x) >> 3) * 5504)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 132096), ((((((((int)blockIdx.x) & 127) * 176128) + ((((int)threadIdx.x) >> 3) * 5504)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 132096));
            int4 __331;
              int4 __332 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __333 = make_int4(2, 2, 2, 2);
              __331.x = (__332.x%__333.x);
              __331.y = (__332.y%__333.y);
              __331.z = (__332.z%__333.z);
              __331.w = (__332.w%__333.w);
            int4 __334;
              int4 __335 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __336 = make_int4(2, 2, 2, 2);
              __334.x = (__335.x/__336.x);
              __334.y = (__335.y/__336.y);
              __334.z = (__335.z/__336.z);
              __334.w = (__335.w/__336.w);
            int4 __337;
            ushort4 __338;
              ushort4 __339;
                ushort4 __340;
                  int4 __341 = make_int4(2, 2, 2, 2);
                  int4 __342 = make_int4(0, 0, 0, 0);
                  __340.x = (__341.x>=__342.x);
                  __340.y = (__341.y>=__342.y);
                  __340.z = (__341.z>=__342.z);
                  __340.w = (__341.w>=__342.w);
                ushort4 __343;
                  int4 __344 = make_int4(0, 0, 0, 0);
                  __343.x = (__331.x>=__344.x);
                  __343.y = (__331.y>=__344.y);
                  __343.z = (__331.z>=__344.z);
                  __343.w = (__331.w>=__344.w);
                __339.x = (__340.x&&__343.x);
                __339.y = (__340.y&&__343.y);
                __339.z = (__340.z&&__343.z);
                __339.w = (__340.w&&__343.w);
              ushort4 __345;
                ushort4 __346;
                  int4 __347 = make_int4(2, 2, 2, 2);
                  int4 __348 = make_int4(0, 0, 0, 0);
                  __346.x = (__347.x<__348.x);
                  __346.y = (__347.y<__348.y);
                  __346.z = (__347.z<__348.z);
                  __346.w = (__347.w<__348.w);
                ushort4 __349;
                  int4 __350 = make_int4(0, 0, 0, 0);
                  __349.x = (__331.x<=__350.x);
                  __349.y = (__331.y<=__350.y);
                  __349.z = (__331.z<=__350.z);
                  __349.w = (__331.w<=__350.w);
                __345.x = (__346.x&&__349.x);
                __345.y = (__346.y&&__349.y);
                __345.z = (__346.z&&__349.z);
                __345.w = (__346.w&&__349.w);
              __338.x = (__339.x||__345.x);
              __338.y = (__339.y||__345.y);
              __338.z = (__339.z||__345.z);
              __338.w = (__339.w||__345.w);
            int4 __351;
              int4 __352 = make_int4(1, 1, 1, 1);
              __351.x = (__334.x-__352.x);
              __351.y = (__334.y-__352.y);
              __351.z = (__334.z-__352.z);
              __351.w = (__334.w-__352.w);
            __337.x = (bool(__338.x)?__334.x:__351.x);
            __337.y = (bool(__338.y)?__334.y:__351.y);
            __337.z = (bool(__338.z)?__334.z:__351.z);
            __337.w = (bool(__338.w)?__334.w:__351.w);
            __329.x = (__330.x+__337.x);
            __329.y = (__330.y+__337.y);
            __329.z = (__330.z+__337.z);
            __329.w = (__330.w+__337.w);
          int __353 = ((0x000000ff << 0) & (B[__329.x] << 0))|((0x000000ff << 8) & (B[__329.y] << 8))|((0x000000ff << 16) & (B[__329.z] << 16))|((0x000000ff << 24) & (B[__329.w] << 24));
          int __354;
          int4 __355;
            int4 __356;
              int4 __357 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __358 = make_int4(2, 2, 2, 2);
              __356.x = (__357.x%__358.x);
              __356.y = (__357.y%__358.y);
              __356.z = (__357.z%__358.z);
              __356.w = (__357.w%__358.w);
            int4 __359;
            ushort4 __360;
              ushort4 __361;
                ushort4 __362;
                  int4 __363 = make_int4(2, 2, 2, 2);
                  int4 __364 = make_int4(0, 0, 0, 0);
                  __362.x = (__363.x>=__364.x);
                  __362.y = (__363.y>=__364.y);
                  __362.z = (__363.z>=__364.z);
                  __362.w = (__363.w>=__364.w);
                ushort4 __365;
                  int4 __366 = make_int4(0, 0, 0, 0);
                  __365.x = (__356.x>=__366.x);
                  __365.y = (__356.y>=__366.y);
                  __365.z = (__356.z>=__366.z);
                  __365.w = (__356.w>=__366.w);
                __361.x = (__362.x&&__365.x);
                __361.y = (__362.y&&__365.y);
                __361.z = (__362.z&&__365.z);
                __361.w = (__362.w&&__365.w);
              ushort4 __367;
                ushort4 __368;
                  int4 __369 = make_int4(2, 2, 2, 2);
                  int4 __370 = make_int4(0, 0, 0, 0);
                  __368.x = (__369.x<__370.x);
                  __368.y = (__369.y<__370.y);
                  __368.z = (__369.z<__370.z);
                  __368.w = (__369.w<__370.w);
                ushort4 __371;
                  int4 __372 = make_int4(0, 0, 0, 0);
                  __371.x = (__356.x<=__372.x);
                  __371.y = (__356.y<=__372.y);
                  __371.z = (__356.z<=__372.z);
                  __371.w = (__356.w<=__372.w);
                __367.x = (__368.x&&__371.x);
                __367.y = (__368.y&&__371.y);
                __367.z = (__368.z&&__371.z);
                __367.w = (__368.w&&__371.w);
              __360.x = (__361.x||__367.x);
              __360.y = (__361.y||__367.y);
              __360.z = (__361.z||__367.z);
              __360.w = (__361.w||__367.w);
            int4 __373;
              int4 __374 = make_int4(2, 2, 2, 2);
              __373.x = (__356.x+__374.x);
              __373.y = (__356.y+__374.y);
              __373.z = (__356.z+__374.z);
              __373.w = (__356.w+__374.w);
            __359.x = (bool(__360.x)?__356.x:__373.x);
            __359.y = (bool(__360.y)?__356.y:__373.y);
            __359.z = (bool(__360.z)?__356.z:__373.z);
            __359.w = (bool(__360.w)?__356.w:__373.w);
            int4 __375 = make_int4(4, 4, 4, 4);
            __355.x = (__359.x*__375.x);
            __355.y = (__359.y*__375.y);
            __355.z = (__359.z*__375.z);
            __355.w = (__359.w*__375.w);
          __354=((signed char)(__355.x) << 0);
          __354=__354 & ~(0x000000ff << 8) |((signed char)(__355.y) << 8);
          __354=__354 & ~(0x000000ff << 16) |((signed char)(__355.z) << 16);
          __354=__354 & ~(0x000000ff << 24) |((signed char)(__355.w) << 24);
          __328=((((char)(__353 >> 0)) >> ((char)(__354 >> 0))) << 0);
          __328=__328 & ~(0x000000ff << 8) |((((char)(__353 >> 8)) >> ((char)(__354 >> 8))) << 8);
          __328=__328 & ~(0x000000ff << 16) |((((char)(__353 >> 16)) >> ((char)(__354 >> 16))) << 16);
          __328=__328 & ~(0x000000ff << 24) |((((char)(__353 >> 24)) >> ((char)(__354 >> 24))) << 24);
        int __376 = (int)252645135;
        __327=((((char)(__328 >> 0)) & ((char)(__376 >> 0))) << 0);
        __327=__327 & ~(0x000000ff << 8) |((((char)(__328 >> 8)) & ((char)(__376 >> 8))) << 8);
        __327=__327 & ~(0x000000ff << 16) |((((char)(__328 >> 16)) & ((char)(__376 >> 16))) << 16);
        __327=__327 & ~(0x000000ff << 24) |((((char)(__328 >> 24)) & ((char)(__376 >> 24))) << 24);
      __326.x = (int)(((char)(__327 >> 0)));
      __326.y = (int)(((char)(__327 >> 8)));
      __326.z = (int)(((char)(__327 >> 16)));
      __326.w = (int)(((char)(__327 >> 24)));
      uint2 __377 = make_uint2(__pack_half2(LUT[__326.x],LUT[__326.y]),__pack_half2(LUT[__326.z],LUT[__326.w]));
      uint2 __378 = make_uint2(__pack_half2(Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 24)], Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 24)]), __pack_half2(Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 24)], Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 24)]));
      ((half2*)(&(__325.x)))->x = (((half2*)(&(__377.x)))->x*((half2*)(&(__378.x)))->x);
      ((half2*)(&(__325.x)))->y = (((half2*)(&(__377.x)))->y*((half2*)(&(__378.x)))->y);
      ((half2*)(&(__325.y)))->x = (((half2*)(&(__377.y)))->x*((half2*)(&(__378.y)))->x);
      ((half2*)(&(__325.y)))->y = (((half2*)(&(__377.y)))->y*((half2*)(&(__378.y)))->y);
    *(uint2*)(B_decode_shared + ((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 4)) + 960)) = __325;
    uint2 __379;
      int4 __380;
      int __381;
        int __382;
          int4 __383;
            int4 __384 = make_int4(((((((((int)blockIdx.x) & 127) * 176128) + ((((int)threadIdx.x) >> 3) * 5504)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 154112), ((((((((int)blockIdx.x) & 127) * 176128) + ((((int)threadIdx.x) >> 3) * 5504)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 154112), ((((((((int)blockIdx.x) & 127) * 176128) + ((((int)threadIdx.x) >> 3) * 5504)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 154112), ((((((((int)blockIdx.x) & 127) * 176128) + ((((int)threadIdx.x) >> 3) * 5504)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 154112));
            int4 __385;
              int4 __386 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __387 = make_int4(2, 2, 2, 2);
              __385.x = (__386.x%__387.x);
              __385.y = (__386.y%__387.y);
              __385.z = (__386.z%__387.z);
              __385.w = (__386.w%__387.w);
            int4 __388;
              int4 __389 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __390 = make_int4(2, 2, 2, 2);
              __388.x = (__389.x/__390.x);
              __388.y = (__389.y/__390.y);
              __388.z = (__389.z/__390.z);
              __388.w = (__389.w/__390.w);
            int4 __391;
            ushort4 __392;
              ushort4 __393;
                ushort4 __394;
                  int4 __395 = make_int4(2, 2, 2, 2);
                  int4 __396 = make_int4(0, 0, 0, 0);
                  __394.x = (__395.x>=__396.x);
                  __394.y = (__395.y>=__396.y);
                  __394.z = (__395.z>=__396.z);
                  __394.w = (__395.w>=__396.w);
                ushort4 __397;
                  int4 __398 = make_int4(0, 0, 0, 0);
                  __397.x = (__385.x>=__398.x);
                  __397.y = (__385.y>=__398.y);
                  __397.z = (__385.z>=__398.z);
                  __397.w = (__385.w>=__398.w);
                __393.x = (__394.x&&__397.x);
                __393.y = (__394.y&&__397.y);
                __393.z = (__394.z&&__397.z);
                __393.w = (__394.w&&__397.w);
              ushort4 __399;
                ushort4 __400;
                  int4 __401 = make_int4(2, 2, 2, 2);
                  int4 __402 = make_int4(0, 0, 0, 0);
                  __400.x = (__401.x<__402.x);
                  __400.y = (__401.y<__402.y);
                  __400.z = (__401.z<__402.z);
                  __400.w = (__401.w<__402.w);
                ushort4 __403;
                  int4 __404 = make_int4(0, 0, 0, 0);
                  __403.x = (__385.x<=__404.x);
                  __403.y = (__385.y<=__404.y);
                  __403.z = (__385.z<=__404.z);
                  __403.w = (__385.w<=__404.w);
                __399.x = (__400.x&&__403.x);
                __399.y = (__400.y&&__403.y);
                __399.z = (__400.z&&__403.z);
                __399.w = (__400.w&&__403.w);
              __392.x = (__393.x||__399.x);
              __392.y = (__393.y||__399.y);
              __392.z = (__393.z||__399.z);
              __392.w = (__393.w||__399.w);
            int4 __405;
              int4 __406 = make_int4(1, 1, 1, 1);
              __405.x = (__388.x-__406.x);
              __405.y = (__388.y-__406.y);
              __405.z = (__388.z-__406.z);
              __405.w = (__388.w-__406.w);
            __391.x = (bool(__392.x)?__388.x:__405.x);
            __391.y = (bool(__392.y)?__388.y:__405.y);
            __391.z = (bool(__392.z)?__388.z:__405.z);
            __391.w = (bool(__392.w)?__388.w:__405.w);
            __383.x = (__384.x+__391.x);
            __383.y = (__384.y+__391.y);
            __383.z = (__384.z+__391.z);
            __383.w = (__384.w+__391.w);
          int __407 = ((0x000000ff << 0) & (B[__383.x] << 0))|((0x000000ff << 8) & (B[__383.y] << 8))|((0x000000ff << 16) & (B[__383.z] << 16))|((0x000000ff << 24) & (B[__383.w] << 24));
          int __408;
          int4 __409;
            int4 __410;
              int4 __411 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __412 = make_int4(2, 2, 2, 2);
              __410.x = (__411.x%__412.x);
              __410.y = (__411.y%__412.y);
              __410.z = (__411.z%__412.z);
              __410.w = (__411.w%__412.w);
            int4 __413;
            ushort4 __414;
              ushort4 __415;
                ushort4 __416;
                  int4 __417 = make_int4(2, 2, 2, 2);
                  int4 __418 = make_int4(0, 0, 0, 0);
                  __416.x = (__417.x>=__418.x);
                  __416.y = (__417.y>=__418.y);
                  __416.z = (__417.z>=__418.z);
                  __416.w = (__417.w>=__418.w);
                ushort4 __419;
                  int4 __420 = make_int4(0, 0, 0, 0);
                  __419.x = (__410.x>=__420.x);
                  __419.y = (__410.y>=__420.y);
                  __419.z = (__410.z>=__420.z);
                  __419.w = (__410.w>=__420.w);
                __415.x = (__416.x&&__419.x);
                __415.y = (__416.y&&__419.y);
                __415.z = (__416.z&&__419.z);
                __415.w = (__416.w&&__419.w);
              ushort4 __421;
                ushort4 __422;
                  int4 __423 = make_int4(2, 2, 2, 2);
                  int4 __424 = make_int4(0, 0, 0, 0);
                  __422.x = (__423.x<__424.x);
                  __422.y = (__423.y<__424.y);
                  __422.z = (__423.z<__424.z);
                  __422.w = (__423.w<__424.w);
                ushort4 __425;
                  int4 __426 = make_int4(0, 0, 0, 0);
                  __425.x = (__410.x<=__426.x);
                  __425.y = (__410.y<=__426.y);
                  __425.z = (__410.z<=__426.z);
                  __425.w = (__410.w<=__426.w);
                __421.x = (__422.x&&__425.x);
                __421.y = (__422.y&&__425.y);
                __421.z = (__422.z&&__425.z);
                __421.w = (__422.w&&__425.w);
              __414.x = (__415.x||__421.x);
              __414.y = (__415.y||__421.y);
              __414.z = (__415.z||__421.z);
              __414.w = (__415.w||__421.w);
            int4 __427;
              int4 __428 = make_int4(2, 2, 2, 2);
              __427.x = (__410.x+__428.x);
              __427.y = (__410.y+__428.y);
              __427.z = (__410.z+__428.z);
              __427.w = (__410.w+__428.w);
            __413.x = (bool(__414.x)?__410.x:__427.x);
            __413.y = (bool(__414.y)?__410.y:__427.y);
            __413.z = (bool(__414.z)?__410.z:__427.z);
            __413.w = (bool(__414.w)?__410.w:__427.w);
            int4 __429 = make_int4(4, 4, 4, 4);
            __409.x = (__413.x*__429.x);
            __409.y = (__413.y*__429.y);
            __409.z = (__413.z*__429.z);
            __409.w = (__413.w*__429.w);
          __408=((signed char)(__409.x) << 0);
          __408=__408 & ~(0x000000ff << 8) |((signed char)(__409.y) << 8);
          __408=__408 & ~(0x000000ff << 16) |((signed char)(__409.z) << 16);
          __408=__408 & ~(0x000000ff << 24) |((signed char)(__409.w) << 24);
          __382=((((char)(__407 >> 0)) >> ((char)(__408 >> 0))) << 0);
          __382=__382 & ~(0x000000ff << 8) |((((char)(__407 >> 8)) >> ((char)(__408 >> 8))) << 8);
          __382=__382 & ~(0x000000ff << 16) |((((char)(__407 >> 16)) >> ((char)(__408 >> 16))) << 16);
          __382=__382 & ~(0x000000ff << 24) |((((char)(__407 >> 24)) >> ((char)(__408 >> 24))) << 24);
        int __430 = (int)252645135;
        __381=((((char)(__382 >> 0)) & ((char)(__430 >> 0))) << 0);
        __381=__381 & ~(0x000000ff << 8) |((((char)(__382 >> 8)) & ((char)(__430 >> 8))) << 8);
        __381=__381 & ~(0x000000ff << 16) |((((char)(__382 >> 16)) & ((char)(__430 >> 16))) << 16);
        __381=__381 & ~(0x000000ff << 24) |((((char)(__382 >> 24)) & ((char)(__430 >> 24))) << 24);
      __380.x = (int)(((char)(__381 >> 0)));
      __380.y = (int)(((char)(__381 >> 8)));
      __380.z = (int)(((char)(__381 >> 16)));
      __380.w = (int)(((char)(__381 >> 24)));
      uint2 __431 = make_uint2(__pack_half2(LUT[__380.x],LUT[__380.y]),__pack_half2(LUT[__380.z],LUT[__380.w]));
      uint2 __432 = make_uint2(__pack_half2(Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 28)], Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 28)]), __pack_half2(Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 28)], Scales[(((((k_outer >> 2) * 4096) + ((((int)blockIdx.x) & 127) * 32)) + (((int)threadIdx.x) >> 3)) + 28)]));
      ((half2*)(&(__379.x)))->x = (((half2*)(&(__431.x)))->x*((half2*)(&(__432.x)))->x);
      ((half2*)(&(__379.x)))->y = (((half2*)(&(__431.x)))->y*((half2*)(&(__432.x)))->y);
      ((half2*)(&(__379.y)))->x = (((half2*)(&(__431.y)))->x*((half2*)(&(__432.y)))->x);
      ((half2*)(&(__379.y)))->y = (((half2*)(&(__431.y)))->y*((half2*)(&(__432.y)))->y);
    *(uint2*)(B_decode_shared + ((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 4)) + 1120)) = __379;
    __syncthreads();
    for (int k_inner_outer = 0; k_inner_outer < 2; ++k_inner_outer) {
      nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[0], (&(A_shared[(k_inner_outer * 16)])), 40);
      nvcuda::wmma::load_matrix_sync(B_decode_shared_wmma_matrix_b[0], (&(B_decode_shared[(k_inner_outer * 16)])), 40);
      nvcuda::wmma::mma_sync(C_wmma_accumulator[0], A_shared_wmma_matrix_a[0], B_decode_shared_wmma_matrix_b[0], C_wmma_accumulator[0]);
    }
  }
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(C[(((((int)blockIdx.x) >> 7) * 32768) + ((((int)blockIdx.x) & 127) * 32))])), C_wmma_accumulator[0], 4096, nvcuda::wmma::mem_row_major);
  __syncthreads();
}



__global__ void __launch_bounds__(128) cutlass_kernel_fp16_nf4_fp16_m16n11008k4096_nt_16x32x32_16x8x16_(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ C) {
            int8_t* B = ((int8_t *)QB);
	 half* LUT = (half *)((int8_t *)QB + 22544384);
	 half* Scales = (half *)((int8_t *)QB + 22544384 + 32);                 
            // const dim3 GridDim(344, 1, 1);
            // const dim3 BlockDim(32, 4, 1);
            // cutlass_kernel_fp16_nf4_fp16_m16n11008k4096_nt_16x32x32_16x8x16_<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
  
  __shared__ half LUT_shared[16];
    __shared__ half A_shared[1024];
  __shared__ half B_decode_shared[2560];
  __shared__ signed char B_shared[1280];
  half B_decode_local[8];
  signed char B_local[2];
  half B_decode_local_1[8];
  signed char B_local_1[2];
  if (((int)threadIdx.x) < 16) {
    LUT_shared[((int)threadIdx.x)] = LUT[((int)threadIdx.x)];
  }
  ALLOCATE_CUTLASS_OBJECT(C_cutlass_warp_mma, cutlass::gemm::warp::GemmTensorOp<
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<16, 32>,
    cutlass::layout::ColumnMajor
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
  #pragma unroll
  for (int ax0_ax1_0_fused_0_0 = 0; ax0_ax1_0_fused_0_0 < 1; ++ax0_ax1_0_fused_0_0) {
    __syncthreads();

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
        :: "r"(addr), "l"((void*)(B + ((((((int)blockIdx.x) * 65536) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 2) * 2048)) + ((((int)threadIdx.x) & 3) * 4)))), "n"(4)
    );
  }
    __syncthreads();
    for (int ax0_0 = 0; ax0_0 < 2; ++ax0_0) {
      *(char2*)(B_local + 0) = *(char2*)(B_shared + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 4)) + (ax0_0 * 2)));
      uint2 __1;
        int4 __2;
        int __3;
          int __4;
            int4 __5;
              int4 __6 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __7 = make_int4(2, 2, 2, 2);
              __5.x = (__6.x%__7.x);
              __5.y = (__6.y%__7.y);
              __5.z = (__6.z%__7.z);
              __5.w = (__6.w%__7.w);
            int4 __8;
              int4 __9 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __10 = make_int4(2, 2, 2, 2);
              __8.x = (__9.x/__10.x);
              __8.y = (__9.y/__10.y);
              __8.z = (__9.z/__10.z);
              __8.w = (__9.w/__10.w);
            int4 __11;
            ushort4 __12;
              ushort4 __13;
                ushort4 __14;
                  int4 __15 = make_int4(2, 2, 2, 2);
                  int4 __16 = make_int4(0, 0, 0, 0);
                  __14.x = (__15.x>=__16.x);
                  __14.y = (__15.y>=__16.y);
                  __14.z = (__15.z>=__16.z);
                  __14.w = (__15.w>=__16.w);
                ushort4 __17;
                  int4 __18 = make_int4(0, 0, 0, 0);
                  __17.x = (__5.x>=__18.x);
                  __17.y = (__5.y>=__18.y);
                  __17.z = (__5.z>=__18.z);
                  __17.w = (__5.w>=__18.w);
                __13.x = (__14.x&&__17.x);
                __13.y = (__14.y&&__17.y);
                __13.z = (__14.z&&__17.z);
                __13.w = (__14.w&&__17.w);
              ushort4 __19;
                ushort4 __20;
                  int4 __21 = make_int4(2, 2, 2, 2);
                  int4 __22 = make_int4(0, 0, 0, 0);
                  __20.x = (__21.x<__22.x);
                  __20.y = (__21.y<__22.y);
                  __20.z = (__21.z<__22.z);
                  __20.w = (__21.w<__22.w);
                ushort4 __23;
                  int4 __24 = make_int4(0, 0, 0, 0);
                  __23.x = (__5.x<=__24.x);
                  __23.y = (__5.y<=__24.y);
                  __23.z = (__5.z<=__24.z);
                  __23.w = (__5.w<=__24.w);
                __19.x = (__20.x&&__23.x);
                __19.y = (__20.y&&__23.y);
                __19.z = (__20.z&&__23.z);
                __19.w = (__20.w&&__23.w);
              __12.x = (__13.x||__19.x);
              __12.y = (__13.y||__19.y);
              __12.z = (__13.z||__19.z);
              __12.w = (__13.w||__19.w);
            int4 __25;
              int4 __26 = make_int4(1, 1, 1, 1);
              __25.x = (__8.x-__26.x);
              __25.y = (__8.y-__26.y);
              __25.z = (__8.z-__26.z);
              __25.w = (__8.w-__26.w);
            __11.x = (bool(__12.x)?__8.x:__25.x);
            __11.y = (bool(__12.y)?__8.y:__25.y);
            __11.z = (bool(__12.z)?__8.z:__25.z);
            __11.w = (bool(__12.w)?__8.w:__25.w);
            int __27 = ((0x000000ff << 0) & (B_local[__11.x] << 0))|((0x000000ff << 8) & (B_local[__11.y] << 8))|((0x000000ff << 16) & (B_local[__11.z] << 16))|((0x000000ff << 24) & (B_local[__11.w] << 24));
            int __28;
            int4 __29;
              int4 __30;
                int4 __31 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
                int4 __32 = make_int4(2, 2, 2, 2);
                __30.x = (__31.x%__32.x);
                __30.y = (__31.y%__32.y);
                __30.z = (__31.z%__32.z);
                __30.w = (__31.w%__32.w);
              int4 __33;
              ushort4 __34;
                ushort4 __35;
                  ushort4 __36;
                    int4 __37 = make_int4(2, 2, 2, 2);
                    int4 __38 = make_int4(0, 0, 0, 0);
                    __36.x = (__37.x>=__38.x);
                    __36.y = (__37.y>=__38.y);
                    __36.z = (__37.z>=__38.z);
                    __36.w = (__37.w>=__38.w);
                  ushort4 __39;
                    int4 __40 = make_int4(0, 0, 0, 0);
                    __39.x = (__30.x>=__40.x);
                    __39.y = (__30.y>=__40.y);
                    __39.z = (__30.z>=__40.z);
                    __39.w = (__30.w>=__40.w);
                  __35.x = (__36.x&&__39.x);
                  __35.y = (__36.y&&__39.y);
                  __35.z = (__36.z&&__39.z);
                  __35.w = (__36.w&&__39.w);
                ushort4 __41;
                  ushort4 __42;
                    int4 __43 = make_int4(2, 2, 2, 2);
                    int4 __44 = make_int4(0, 0, 0, 0);
                    __42.x = (__43.x<__44.x);
                    __42.y = (__43.y<__44.y);
                    __42.z = (__43.z<__44.z);
                    __42.w = (__43.w<__44.w);
                  ushort4 __45;
                    int4 __46 = make_int4(0, 0, 0, 0);
                    __45.x = (__30.x<=__46.x);
                    __45.y = (__30.y<=__46.y);
                    __45.z = (__30.z<=__46.z);
                    __45.w = (__30.w<=__46.w);
                  __41.x = (__42.x&&__45.x);
                  __41.y = (__42.y&&__45.y);
                  __41.z = (__42.z&&__45.z);
                  __41.w = (__42.w&&__45.w);
                __34.x = (__35.x||__41.x);
                __34.y = (__35.y||__41.y);
                __34.z = (__35.z||__41.z);
                __34.w = (__35.w||__41.w);
              int4 __47;
                int4 __48 = make_int4(2, 2, 2, 2);
                __47.x = (__30.x+__48.x);
                __47.y = (__30.y+__48.y);
                __47.z = (__30.z+__48.z);
                __47.w = (__30.w+__48.w);
              __33.x = (bool(__34.x)?__30.x:__47.x);
              __33.y = (bool(__34.y)?__30.y:__47.y);
              __33.z = (bool(__34.z)?__30.z:__47.z);
              __33.w = (bool(__34.w)?__30.w:__47.w);
              int4 __49 = make_int4(4, 4, 4, 4);
              __29.x = (__33.x*__49.x);
              __29.y = (__33.y*__49.y);
              __29.z = (__33.z*__49.z);
              __29.w = (__33.w*__49.w);
            __28=((signed char)(__29.x) << 0);
            __28=__28 & ~(0x000000ff << 8) |((signed char)(__29.y) << 8);
            __28=__28 & ~(0x000000ff << 16) |((signed char)(__29.z) << 16);
            __28=__28 & ~(0x000000ff << 24) |((signed char)(__29.w) << 24);
            __4=((((char)(__27 >> 0)) >> ((char)(__28 >> 0))) << 0);
            __4=__4 & ~(0x000000ff << 8) |((((char)(__27 >> 8)) >> ((char)(__28 >> 8))) << 8);
            __4=__4 & ~(0x000000ff << 16) |((((char)(__27 >> 16)) >> ((char)(__28 >> 16))) << 16);
            __4=__4 & ~(0x000000ff << 24) |((((char)(__27 >> 24)) >> ((char)(__28 >> 24))) << 24);
          int __50 = (int)252645135;
          __3=((((char)(__4 >> 0)) & ((char)(__50 >> 0))) << 0);
          __3=__3 & ~(0x000000ff << 8) |((((char)(__4 >> 8)) & ((char)(__50 >> 8))) << 8);
          __3=__3 & ~(0x000000ff << 16) |((((char)(__4 >> 16)) & ((char)(__50 >> 16))) << 16);
          __3=__3 & ~(0x000000ff << 24) |((((char)(__4 >> 24)) & ((char)(__50 >> 24))) << 24);
        __2.x = (int)(((char)(__3 >> 0)));
        __2.y = (int)(((char)(__3 >> 8)));
        __2.z = (int)(((char)(__3 >> 16)));
        __2.w = (int)(((char)(__3 >> 24)));
        uint2 __51 = make_uint2(__pack_half2(LUT_shared[__2.x],LUT_shared[__2.y]),__pack_half2(LUT_shared[__2.z],LUT_shared[__2.w]));
        uint2 __52 = make_uint2(__pack_half2(Scales[(((((int)blockIdx.x) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))], Scales[(((((int)blockIdx.x) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))]), __pack_half2(Scales[(((((int)blockIdx.x) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))], Scales[(((((int)blockIdx.x) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))]));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(__51.x)))->x*((half2*)(&(__52.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(__51.x)))->y*((half2*)(&(__52.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(__51.y)))->x*((half2*)(&(__52.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(__51.y)))->y*((half2*)(&(__52.y)))->y);
      *(uint2*)(B_decode_local + (ax0_0 * 4)) = __1;
    }
    *(uint4*)(B_decode_shared + (((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) = *(uint4*)(B_decode_local + 0);
  }
__asm__ __volatile__("cp.async.commit_group;");

  #pragma unroll
  for (int ax0_ax1_fused_0_0_0_1 = 0; ax0_ax1_fused_0_0_0_1 < 1; ++ax0_ax1_fused_0_0_0_1) {
    if (((int)threadIdx.y) < 2) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + (((((((int)threadIdx.y) * 256) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 512))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(A_shared + (((((((int)threadIdx.y) * 256) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 512)))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((int)threadIdx.y) * 32768) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 32))), "n"(16)
    );
  }
    }
  }
  #pragma unroll
  for (int ax0_ax1_0_fused_0_0_1 = 0; ax0_ax1_0_fused_0_0_1 < 1; ++ax0_ax1_0_fused_0_0_1) {
    __syncthreads();

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
        :: "r"(addr), "l"((void*)(B + (((((((int)blockIdx.x) * 65536) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 2) * 2048)) + ((((int)threadIdx.x) & 3) * 4)) + 16))), "n"(4)
    );
  }
    __syncthreads();
    for (int ax0_0_1 = 0; ax0_0_1 < 2; ++ax0_0_1) {
      *(char2*)(B_local + 0) = *(char2*)(B_shared + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 4)) + (ax0_0_1 * 2)));
      uint2 __53;
        int4 __54;
        int __55;
          int __56;
            int4 __57;
              int4 __58 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __59 = make_int4(2, 2, 2, 2);
              __57.x = (__58.x%__59.x);
              __57.y = (__58.y%__59.y);
              __57.z = (__58.z%__59.z);
              __57.w = (__58.w%__59.w);
            int4 __60;
              int4 __61 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __62 = make_int4(2, 2, 2, 2);
              __60.x = (__61.x/__62.x);
              __60.y = (__61.y/__62.y);
              __60.z = (__61.z/__62.z);
              __60.w = (__61.w/__62.w);
            int4 __63;
            ushort4 __64;
              ushort4 __65;
                ushort4 __66;
                  int4 __67 = make_int4(2, 2, 2, 2);
                  int4 __68 = make_int4(0, 0, 0, 0);
                  __66.x = (__67.x>=__68.x);
                  __66.y = (__67.y>=__68.y);
                  __66.z = (__67.z>=__68.z);
                  __66.w = (__67.w>=__68.w);
                ushort4 __69;
                  int4 __70 = make_int4(0, 0, 0, 0);
                  __69.x = (__57.x>=__70.x);
                  __69.y = (__57.y>=__70.y);
                  __69.z = (__57.z>=__70.z);
                  __69.w = (__57.w>=__70.w);
                __65.x = (__66.x&&__69.x);
                __65.y = (__66.y&&__69.y);
                __65.z = (__66.z&&__69.z);
                __65.w = (__66.w&&__69.w);
              ushort4 __71;
                ushort4 __72;
                  int4 __73 = make_int4(2, 2, 2, 2);
                  int4 __74 = make_int4(0, 0, 0, 0);
                  __72.x = (__73.x<__74.x);
                  __72.y = (__73.y<__74.y);
                  __72.z = (__73.z<__74.z);
                  __72.w = (__73.w<__74.w);
                ushort4 __75;
                  int4 __76 = make_int4(0, 0, 0, 0);
                  __75.x = (__57.x<=__76.x);
                  __75.y = (__57.y<=__76.y);
                  __75.z = (__57.z<=__76.z);
                  __75.w = (__57.w<=__76.w);
                __71.x = (__72.x&&__75.x);
                __71.y = (__72.y&&__75.y);
                __71.z = (__72.z&&__75.z);
                __71.w = (__72.w&&__75.w);
              __64.x = (__65.x||__71.x);
              __64.y = (__65.y||__71.y);
              __64.z = (__65.z||__71.z);
              __64.w = (__65.w||__71.w);
            int4 __77;
              int4 __78 = make_int4(1, 1, 1, 1);
              __77.x = (__60.x-__78.x);
              __77.y = (__60.y-__78.y);
              __77.z = (__60.z-__78.z);
              __77.w = (__60.w-__78.w);
            __63.x = (bool(__64.x)?__60.x:__77.x);
            __63.y = (bool(__64.y)?__60.y:__77.y);
            __63.z = (bool(__64.z)?__60.z:__77.z);
            __63.w = (bool(__64.w)?__60.w:__77.w);
            int __79 = ((0x000000ff << 0) & (B_local[__63.x] << 0))|((0x000000ff << 8) & (B_local[__63.y] << 8))|((0x000000ff << 16) & (B_local[__63.z] << 16))|((0x000000ff << 24) & (B_local[__63.w] << 24));
            int __80;
            int4 __81;
              int4 __82;
                int4 __83 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
                int4 __84 = make_int4(2, 2, 2, 2);
                __82.x = (__83.x%__84.x);
                __82.y = (__83.y%__84.y);
                __82.z = (__83.z%__84.z);
                __82.w = (__83.w%__84.w);
              int4 __85;
              ushort4 __86;
                ushort4 __87;
                  ushort4 __88;
                    int4 __89 = make_int4(2, 2, 2, 2);
                    int4 __90 = make_int4(0, 0, 0, 0);
                    __88.x = (__89.x>=__90.x);
                    __88.y = (__89.y>=__90.y);
                    __88.z = (__89.z>=__90.z);
                    __88.w = (__89.w>=__90.w);
                  ushort4 __91;
                    int4 __92 = make_int4(0, 0, 0, 0);
                    __91.x = (__82.x>=__92.x);
                    __91.y = (__82.y>=__92.y);
                    __91.z = (__82.z>=__92.z);
                    __91.w = (__82.w>=__92.w);
                  __87.x = (__88.x&&__91.x);
                  __87.y = (__88.y&&__91.y);
                  __87.z = (__88.z&&__91.z);
                  __87.w = (__88.w&&__91.w);
                ushort4 __93;
                  ushort4 __94;
                    int4 __95 = make_int4(2, 2, 2, 2);
                    int4 __96 = make_int4(0, 0, 0, 0);
                    __94.x = (__95.x<__96.x);
                    __94.y = (__95.y<__96.y);
                    __94.z = (__95.z<__96.z);
                    __94.w = (__95.w<__96.w);
                  ushort4 __97;
                    int4 __98 = make_int4(0, 0, 0, 0);
                    __97.x = (__82.x<=__98.x);
                    __97.y = (__82.y<=__98.y);
                    __97.z = (__82.z<=__98.z);
                    __97.w = (__82.w<=__98.w);
                  __93.x = (__94.x&&__97.x);
                  __93.y = (__94.y&&__97.y);
                  __93.z = (__94.z&&__97.z);
                  __93.w = (__94.w&&__97.w);
                __86.x = (__87.x||__93.x);
                __86.y = (__87.y||__93.y);
                __86.z = (__87.z||__93.z);
                __86.w = (__87.w||__93.w);
              int4 __99;
                int4 __100 = make_int4(2, 2, 2, 2);
                __99.x = (__82.x+__100.x);
                __99.y = (__82.y+__100.y);
                __99.z = (__82.z+__100.z);
                __99.w = (__82.w+__100.w);
              __85.x = (bool(__86.x)?__82.x:__99.x);
              __85.y = (bool(__86.y)?__82.y:__99.y);
              __85.z = (bool(__86.z)?__82.z:__99.z);
              __85.w = (bool(__86.w)?__82.w:__99.w);
              int4 __101 = make_int4(4, 4, 4, 4);
              __81.x = (__85.x*__101.x);
              __81.y = (__85.y*__101.y);
              __81.z = (__85.z*__101.z);
              __81.w = (__85.w*__101.w);
            __80=((signed char)(__81.x) << 0);
            __80=__80 & ~(0x000000ff << 8) |((signed char)(__81.y) << 8);
            __80=__80 & ~(0x000000ff << 16) |((signed char)(__81.z) << 16);
            __80=__80 & ~(0x000000ff << 24) |((signed char)(__81.w) << 24);
            __56=((((char)(__79 >> 0)) >> ((char)(__80 >> 0))) << 0);
            __56=__56 & ~(0x000000ff << 8) |((((char)(__79 >> 8)) >> ((char)(__80 >> 8))) << 8);
            __56=__56 & ~(0x000000ff << 16) |((((char)(__79 >> 16)) >> ((char)(__80 >> 16))) << 16);
            __56=__56 & ~(0x000000ff << 24) |((((char)(__79 >> 24)) >> ((char)(__80 >> 24))) << 24);
          int __102 = (int)252645135;
          __55=((((char)(__56 >> 0)) & ((char)(__102 >> 0))) << 0);
          __55=__55 & ~(0x000000ff << 8) |((((char)(__56 >> 8)) & ((char)(__102 >> 8))) << 8);
          __55=__55 & ~(0x000000ff << 16) |((((char)(__56 >> 16)) & ((char)(__102 >> 16))) << 16);
          __55=__55 & ~(0x000000ff << 24) |((((char)(__56 >> 24)) & ((char)(__102 >> 24))) << 24);
        __54.x = (int)(((char)(__55 >> 0)));
        __54.y = (int)(((char)(__55 >> 8)));
        __54.z = (int)(((char)(__55 >> 16)));
        __54.w = (int)(((char)(__55 >> 24)));
        uint2 __103 = make_uint2(__pack_half2(LUT_shared[__54.x],LUT_shared[__54.y]),__pack_half2(LUT_shared[__54.z],LUT_shared[__54.w]));
        uint2 __104 = make_uint2(__pack_half2(Scales[(((((int)blockIdx.x) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))], Scales[(((((int)blockIdx.x) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))]), __pack_half2(Scales[(((((int)blockIdx.x) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))], Scales[(((((int)blockIdx.x) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))]));
        ((half2*)(&(__53.x)))->x = (((half2*)(&(__103.x)))->x*((half2*)(&(__104.x)))->x);
        ((half2*)(&(__53.x)))->y = (((half2*)(&(__103.x)))->y*((half2*)(&(__104.x)))->y);
        ((half2*)(&(__53.y)))->x = (((half2*)(&(__103.y)))->x*((half2*)(&(__104.y)))->x);
        ((half2*)(&(__53.y)))->y = (((half2*)(&(__103.y)))->y*((half2*)(&(__104.y)))->y);
      *(uint2*)(B_decode_local + (ax0_0_1 * 4)) = __53;
    }
    *(uint4*)(B_decode_shared + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 1280)) = *(uint4*)(B_decode_local + 0);
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 1;");

  __syncthreads();
  call_cutlass_mma_prologue(C_cutlass_warp_mma, (&(A_shared[0])), (&(B_decode_shared[0])), 32, 40);
  call_cutlass_mma_body(C_cutlass_warp_mma);
  for (int k_0 = 0; k_0 < 126; ++k_0) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_0_0_0_2 = 0; ax0_ax1_fused_0_0_0_2 < 1; ++ax0_ax1_fused_0_0_0_2) {
      if (((int)threadIdx.y) < 2) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + ((((((k_0 & 1) * 512) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(A_shared + ((((((k_0 & 1) * 512) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + (((((((int)threadIdx.y) * 32768) + ((((int)threadIdx.x) >> 2) * 4096)) + (k_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 64))), "n"(16)
    );
  }
      }
    }
    #pragma unroll
    for (int ax0_ax1_0_fused_0_0_2 = 0; ax0_ax1_0_fused_0_0_2 < 1; ++ax0_ax1_0_fused_0_0_2) {
      __syncthreads();

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
        :: "r"(addr), "l"((void*)(B + ((((((((int)blockIdx.x) * 65536) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 2) * 2048)) + (k_0 * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 32))), "n"(4)
    );
  }
      __syncthreads();
      for (int ax0_0_2 = 0; ax0_0_2 < 2; ++ax0_0_2) {
        *(char2*)(B_local_1 + 0) = *(char2*)(B_shared + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 4)) + (ax0_0_2 * 2)));
        uint2 __105;
          int4 __106;
          int __107;
            int __108;
              int4 __109;
                int4 __110 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
                int4 __111 = make_int4(2, 2, 2, 2);
                __109.x = (__110.x%__111.x);
                __109.y = (__110.y%__111.y);
                __109.z = (__110.z%__111.z);
                __109.w = (__110.w%__111.w);
              int4 __112;
                int4 __113 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
                int4 __114 = make_int4(2, 2, 2, 2);
                __112.x = (__113.x/__114.x);
                __112.y = (__113.y/__114.y);
                __112.z = (__113.z/__114.z);
                __112.w = (__113.w/__114.w);
              int4 __115;
              ushort4 __116;
                ushort4 __117;
                  ushort4 __118;
                    int4 __119 = make_int4(2, 2, 2, 2);
                    int4 __120 = make_int4(0, 0, 0, 0);
                    __118.x = (__119.x>=__120.x);
                    __118.y = (__119.y>=__120.y);
                    __118.z = (__119.z>=__120.z);
                    __118.w = (__119.w>=__120.w);
                  ushort4 __121;
                    int4 __122 = make_int4(0, 0, 0, 0);
                    __121.x = (__109.x>=__122.x);
                    __121.y = (__109.y>=__122.y);
                    __121.z = (__109.z>=__122.z);
                    __121.w = (__109.w>=__122.w);
                  __117.x = (__118.x&&__121.x);
                  __117.y = (__118.y&&__121.y);
                  __117.z = (__118.z&&__121.z);
                  __117.w = (__118.w&&__121.w);
                ushort4 __123;
                  ushort4 __124;
                    int4 __125 = make_int4(2, 2, 2, 2);
                    int4 __126 = make_int4(0, 0, 0, 0);
                    __124.x = (__125.x<__126.x);
                    __124.y = (__125.y<__126.y);
                    __124.z = (__125.z<__126.z);
                    __124.w = (__125.w<__126.w);
                  ushort4 __127;
                    int4 __128 = make_int4(0, 0, 0, 0);
                    __127.x = (__109.x<=__128.x);
                    __127.y = (__109.y<=__128.y);
                    __127.z = (__109.z<=__128.z);
                    __127.w = (__109.w<=__128.w);
                  __123.x = (__124.x&&__127.x);
                  __123.y = (__124.y&&__127.y);
                  __123.z = (__124.z&&__127.z);
                  __123.w = (__124.w&&__127.w);
                __116.x = (__117.x||__123.x);
                __116.y = (__117.y||__123.y);
                __116.z = (__117.z||__123.z);
                __116.w = (__117.w||__123.w);
              int4 __129;
                int4 __130 = make_int4(1, 1, 1, 1);
                __129.x = (__112.x-__130.x);
                __129.y = (__112.y-__130.y);
                __129.z = (__112.z-__130.z);
                __129.w = (__112.w-__130.w);
              __115.x = (bool(__116.x)?__112.x:__129.x);
              __115.y = (bool(__116.y)?__112.y:__129.y);
              __115.z = (bool(__116.z)?__112.z:__129.z);
              __115.w = (bool(__116.w)?__112.w:__129.w);
              int __131 = ((0x000000ff << 0) & (B_local_1[__115.x] << 0))|((0x000000ff << 8) & (B_local_1[__115.y] << 8))|((0x000000ff << 16) & (B_local_1[__115.z] << 16))|((0x000000ff << 24) & (B_local_1[__115.w] << 24));
              int __132;
              int4 __133;
                int4 __134;
                  int4 __135 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
                  int4 __136 = make_int4(2, 2, 2, 2);
                  __134.x = (__135.x%__136.x);
                  __134.y = (__135.y%__136.y);
                  __134.z = (__135.z%__136.z);
                  __134.w = (__135.w%__136.w);
                int4 __137;
                ushort4 __138;
                  ushort4 __139;
                    ushort4 __140;
                      int4 __141 = make_int4(2, 2, 2, 2);
                      int4 __142 = make_int4(0, 0, 0, 0);
                      __140.x = (__141.x>=__142.x);
                      __140.y = (__141.y>=__142.y);
                      __140.z = (__141.z>=__142.z);
                      __140.w = (__141.w>=__142.w);
                    ushort4 __143;
                      int4 __144 = make_int4(0, 0, 0, 0);
                      __143.x = (__134.x>=__144.x);
                      __143.y = (__134.y>=__144.y);
                      __143.z = (__134.z>=__144.z);
                      __143.w = (__134.w>=__144.w);
                    __139.x = (__140.x&&__143.x);
                    __139.y = (__140.y&&__143.y);
                    __139.z = (__140.z&&__143.z);
                    __139.w = (__140.w&&__143.w);
                  ushort4 __145;
                    ushort4 __146;
                      int4 __147 = make_int4(2, 2, 2, 2);
                      int4 __148 = make_int4(0, 0, 0, 0);
                      __146.x = (__147.x<__148.x);
                      __146.y = (__147.y<__148.y);
                      __146.z = (__147.z<__148.z);
                      __146.w = (__147.w<__148.w);
                    ushort4 __149;
                      int4 __150 = make_int4(0, 0, 0, 0);
                      __149.x = (__134.x<=__150.x);
                      __149.y = (__134.y<=__150.y);
                      __149.z = (__134.z<=__150.z);
                      __149.w = (__134.w<=__150.w);
                    __145.x = (__146.x&&__149.x);
                    __145.y = (__146.y&&__149.y);
                    __145.z = (__146.z&&__149.z);
                    __145.w = (__146.w&&__149.w);
                  __138.x = (__139.x||__145.x);
                  __138.y = (__139.y||__145.y);
                  __138.z = (__139.z||__145.z);
                  __138.w = (__139.w||__145.w);
                int4 __151;
                  int4 __152 = make_int4(2, 2, 2, 2);
                  __151.x = (__134.x+__152.x);
                  __151.y = (__134.y+__152.y);
                  __151.z = (__134.z+__152.z);
                  __151.w = (__134.w+__152.w);
                __137.x = (bool(__138.x)?__134.x:__151.x);
                __137.y = (bool(__138.y)?__134.y:__151.y);
                __137.z = (bool(__138.z)?__134.z:__151.z);
                __137.w = (bool(__138.w)?__134.w:__151.w);
                int4 __153 = make_int4(4, 4, 4, 4);
                __133.x = (__137.x*__153.x);
                __133.y = (__137.y*__153.y);
                __133.z = (__137.z*__153.z);
                __133.w = (__137.w*__153.w);
              __132=((signed char)(__133.x) << 0);
              __132=__132 & ~(0x000000ff << 8) |((signed char)(__133.y) << 8);
              __132=__132 & ~(0x000000ff << 16) |((signed char)(__133.z) << 16);
              __132=__132 & ~(0x000000ff << 24) |((signed char)(__133.w) << 24);
              __108=((((char)(__131 >> 0)) >> ((char)(__132 >> 0))) << 0);
              __108=__108 & ~(0x000000ff << 8) |((((char)(__131 >> 8)) >> ((char)(__132 >> 8))) << 8);
              __108=__108 & ~(0x000000ff << 16) |((((char)(__131 >> 16)) >> ((char)(__132 >> 16))) << 16);
              __108=__108 & ~(0x000000ff << 24) |((((char)(__131 >> 24)) >> ((char)(__132 >> 24))) << 24);
            int __154 = (int)252645135;
            __107=((((char)(__108 >> 0)) & ((char)(__154 >> 0))) << 0);
            __107=__107 & ~(0x000000ff << 8) |((((char)(__108 >> 8)) & ((char)(__154 >> 8))) << 8);
            __107=__107 & ~(0x000000ff << 16) |((((char)(__108 >> 16)) & ((char)(__154 >> 16))) << 16);
            __107=__107 & ~(0x000000ff << 24) |((((char)(__108 >> 24)) & ((char)(__154 >> 24))) << 24);
          __106.x = (int)(((char)(__107 >> 0)));
          __106.y = (int)(((char)(__107 >> 8)));
          __106.z = (int)(((char)(__107 >> 16)));
          __106.w = (int)(((char)(__107 >> 24)));
          uint2 __155 = make_uint2(__pack_half2(LUT_shared[__106.x],LUT_shared[__106.y]),__pack_half2(LUT_shared[__106.z],LUT_shared[__106.w]));
          int4 __156 = make_int4((((((((((k_0 * 32) + ((((int)threadIdx.x) & 3) * 8)) + (ax0_0_2 * 4)) + 64) >> 7) * 11008) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)), (((((((((k_0 * 32) + ((((int)threadIdx.x) & 3) * 8)) + (ax0_0_2 * 4)) + 64) >> 7) * 11008) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)), (((((((((k_0 * 32) + ((((int)threadIdx.x) & 3) * 8)) + (ax0_0_2 * 4)) + 64) >> 7) * 11008) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)), (((((((((k_0 * 32) + ((((int)threadIdx.x) & 3) * 8)) + (ax0_0_2 * 4)) + 64) >> 7) * 11008) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)));
          uint2 __157 = make_uint2(__pack_half2(Scales[__156.x],Scales[__156.y]),__pack_half2(Scales[__156.z],Scales[__156.w]));
          ((half2*)(&(__105.x)))->x = (((half2*)(&(__155.x)))->x*((half2*)(&(__157.x)))->x);
          ((half2*)(&(__105.x)))->y = (((half2*)(&(__155.x)))->y*((half2*)(&(__157.x)))->y);
          ((half2*)(&(__105.y)))->x = (((half2*)(&(__155.y)))->x*((half2*)(&(__157.y)))->x);
          ((half2*)(&(__105.y)))->y = (((half2*)(&(__155.y)))->y*((half2*)(&(__157.y)))->y);
        *(uint2*)(B_decode_local_1 + (ax0_0_2 * 4)) = __105;
      }
      *(uint4*)(B_decode_shared + (((((k_0 & 1) * 1280) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) = *(uint4*)(B_decode_local_1 + 0);
    }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 1;");

    __syncthreads();
    call_cutlass_mma_prologue(C_cutlass_warp_mma, (&(A_shared[(((k_0 + 1) & 1) * 512)])), (&(B_decode_shared[(((k_0 + 1) & 1) * 1280)])), 32, 40);
    call_cutlass_mma_epilogue(C_cutlass_warp_mma);
    call_cutlass_mma_body(C_cutlass_warp_mma);
  }
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  call_cutlass_mma_prologue(C_cutlass_warp_mma, (&(A_shared[512])), (&(B_decode_shared[1280])), 32, 40);
  call_cutlass_mma_epilogue(C_cutlass_warp_mma);
  call_cutlass_mma_body(C_cutlass_warp_mma);
  call_cutlass_mma_epilogue(C_cutlass_warp_mma);
  #pragma unroll
  for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0) {
    *(uint1*)(C + (((((ax1_0 * 88064) + ((((int)threadIdx.x) >> 2) * 11008)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.y) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(C_cutlass_warp_mma + (ax1_0 * 2));
  }
}



 __global__ void __launch_bounds__(128) cutlass_kernel_fp16_nf4_fp16_m1n12288k4096_nt_4x8(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ C) {
            int8_t* B = ((int8_t *)QB);
	 half* LUT = (half *)((int8_t *)QB + 25165824);
	 half* Scales = (half *)((int8_t *)QB + 25165824 + 32);                 
            // const dim3 GridDim(3072, 1, 1);
            // const dim3 BlockDim(32, 4, 1);
            // cutlass_kernel_fp16_nf4_fp16_m1n12288k4096_nt_4x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
  half in_thread_C_local[1];
  half A_local[8];
  signed char B_local[4];
  half red_buf0[1];
  in_thread_C_local[0] = __float2half_rn(0.000000e+00f);
  for (int k_0 = 0; k_0 < 16; ++k_0) {
    *(uint4*)(A_local + 0) = *(uint4*)(A + ((k_0 * 256) + (((int)threadIdx.x) * 8)));
    *(int*)(B_local + 0) = *(int*)(B + ((((((int)blockIdx.x) * 8192) + (((int)threadIdx.y) * 2048)) + (k_0 * 128)) + (((int)threadIdx.x) * 4)));
    for (int k_2 = 0; k_2 < 8; ++k_2) {
      in_thread_C_local[0] = (in_thread_C_local[0] + (A_local[k_2] * (LUT[((int)((B_local[(k_2 >> 1)] >> ((signed char)((k_2 & 1) * 4))) & (signed char)15))] * Scales[((((k_0 * 24576) + ((((int)threadIdx.x) >> 4) * 12288)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.y))])));
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



 __global__ void __launch_bounds__(128) cutlass_kernel_fp16_nf4_fp16_m1n4096k4096_nt_4x8(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ C) {
            int8_t* B = ((int8_t *)QB);
	 half* LUT = (half *)((int8_t *)QB + 8388608);
	 half* Scales = (half *)((int8_t *)QB + 8388608 + 32);                 
            // const dim3 GridDim(1024, 1, 1);
            // const dim3 BlockDim(32, 4, 1);
            // cutlass_kernel_fp16_nf4_fp16_m1n4096k4096_nt_4x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
  half in_thread_C_local[1];
  half A_local[8];
  signed char B_local[4];
  half red_buf0[1];
  in_thread_C_local[0] = __float2half_rn(0.000000e+00f);
  for (int k_0 = 0; k_0 < 16; ++k_0) {
    *(uint4*)(A_local + 0) = *(uint4*)(A + ((k_0 * 256) + (((int)threadIdx.x) * 8)));
    *(int*)(B_local + 0) = *(int*)(B + ((((((int)blockIdx.x) * 8192) + (((int)threadIdx.y) * 2048)) + (k_0 * 128)) + (((int)threadIdx.x) * 4)));
    for (int k_2 = 0; k_2 < 8; ++k_2) {
      in_thread_C_local[0] = (in_thread_C_local[0] + (A_local[k_2] * (LUT[((int)((B_local[(k_2 >> 1)] >> ((signed char)((k_2 & 1) * 4))) & (signed char)15))] * Scales[((((k_0 * 8192) + ((((int)threadIdx.x) >> 4) * 4096)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.y))])));
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



 __global__ void __launch_bounds__(128) cutlass_kernel_fp16_nf4_fp16_m1n4096k11008_nt_4x8(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ C) {
            int8_t* B = ((int8_t *)QB);
	 half* LUT = (half *)((int8_t *)QB + 22544384);
	 half* Scales = (half *)((int8_t *)QB + 22544384 + 32);                 
            // const dim3 GridDim(1024, 1, 1);
            // const dim3 BlockDim(32, 4, 1);
            // cutlass_kernel_fp16_nf4_fp16_m1n4096k11008_nt_4x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
  half in_thread_C_local[1];
  half A_local[8];
  signed char B_local[4];
  half red_buf0[1];
  in_thread_C_local[0] = __float2half_rn(0.000000e+00f);
  for (int k_0 = 0; k_0 < 43; ++k_0) {
    *(uint4*)(A_local + 0) = *(uint4*)(A + ((k_0 * 256) + (((int)threadIdx.x) * 8)));
    *(int*)(B_local + 0) = *(int*)(B + ((((((int)blockIdx.x) * 22016) + (((int)threadIdx.y) * 5504)) + (k_0 * 128)) + (((int)threadIdx.x) * 4)));
    for (int k_2 = 0; k_2 < 8; ++k_2) {
      in_thread_C_local[0] = (in_thread_C_local[0] + (A_local[k_2] * (LUT[((int)((B_local[(k_2 >> 1)] >> ((signed char)((k_2 & 1) * 4))) & (signed char)15))] * Scales[((((k_0 * 8192) + ((((int)threadIdx.x) >> 4) * 4096)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.y))])));
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



 __global__ void __launch_bounds__(128) cutlass_kernel_fp16_nf4_fp16_m1n11008k4096_nt_4x8(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ C) {
            int8_t* B = ((int8_t *)QB);
	 half* LUT = (half *)((int8_t *)QB + 22544384);
	 half* Scales = (half *)((int8_t *)QB + 22544384 + 32);                 
            // const dim3 GridDim(2752, 1, 1);
            // const dim3 BlockDim(32, 4, 1);
            // cutlass_kernel_fp16_nf4_fp16_m1n11008k4096_nt_4x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
  half in_thread_C_local[1];
  half A_local[8];
  signed char B_local[4];
  half red_buf0[1];
  in_thread_C_local[0] = __float2half_rn(0.000000e+00f);
  for (int k_0 = 0; k_0 < 16; ++k_0) {
    *(uint4*)(A_local + 0) = *(uint4*)(A + ((k_0 * 256) + (((int)threadIdx.x) * 8)));
    *(int*)(B_local + 0) = *(int*)(B + ((((((int)blockIdx.x) * 8192) + (((int)threadIdx.y) * 2048)) + (k_0 * 128)) + (((int)threadIdx.x) * 4)));
    for (int k_2 = 0; k_2 < 8; ++k_2) {
      in_thread_C_local[0] = (in_thread_C_local[0] + (A_local[k_2] * (LUT[((int)((B_local[(k_2 >> 1)] >> ((signed char)((k_2 & 1) * 4))) & (signed char)15))] * Scales[((((k_0 * 22016) + ((((int)threadIdx.x) >> 4) * 11008)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.y))])));
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



__global__ void __launch_bounds__(96) cutlass_kernel_fp16_nf4_fp16_m16n15360k5120_nt_16x48x32_16x8x16_(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ C) {
            int8_t* B = ((int8_t *)QB);
	 half* LUT = (half *)((int8_t *)QB + 39321600);
	 half* Scales = (half *)((int8_t *)QB + 39321600 + 32);                 
            // const dim3 GridDim(320, 1, 1);
            // const dim3 BlockDim(32, 3, 1);
            // cutlass_kernel_fp16_nf4_fp16_m16n15360k5120_nt_16x48x32_16x8x16_<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
  
  __shared__ half LUT_shared[16];
    __shared__ half A_shared[1024];
  __shared__ half B_decode_shared[3072];
  __shared__ signed char B_shared[384];
  half B_decode_local[8];
  signed char B_local[2];
  half B_decode_local_1[8];
  signed char B_local_1[2];
  if (((int)threadIdx.x) < 16) {
    LUT_shared[((int)threadIdx.x)] = LUT[((int)threadIdx.x)];
  }
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
        :: "r"(addr), "l"((void*)(A + (((((int)threadIdx.y) * 40960) + ((((int)threadIdx.x) >> 2) * 5120)) + ((((int)threadIdx.x) & 3) * 8)))), "n"(16)
    );
  }
    }
  }
  #pragma unroll
  for (int ax0_ax1_0_fused_0_0 = 0; ax0_ax1_0_fused_0_0 < 2; ++ax0_ax1_0_fused_0_0) {
    __syncthreads();

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(B_shared + ((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(B_shared + ((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.ca.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + (((((((int)blockIdx.x) * 122880) + (ax0_ax1_0_fused_0_0 * 61440)) + (((int)threadIdx.y) * 20480)) + ((((int)threadIdx.x) >> 2) * 2560)) + ((((int)threadIdx.x) & 3) * 4)))), "n"(4)
    );
  }
    __syncthreads();
    for (int ax0_0 = 0; ax0_0 < 2; ++ax0_0) {
      *(char2*)(B_local + 0) = *(char2*)(B_shared + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + (ax0_0 * 2)));
      uint2 __1;
        int4 __2;
        int __3;
          int __4;
            int4 __5;
              int4 __6 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __7 = make_int4(2, 2, 2, 2);
              __5.x = (__6.x%__7.x);
              __5.y = (__6.y%__7.y);
              __5.z = (__6.z%__7.z);
              __5.w = (__6.w%__7.w);
            int4 __8;
              int4 __9 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __10 = make_int4(2, 2, 2, 2);
              __8.x = (__9.x/__10.x);
              __8.y = (__9.y/__10.y);
              __8.z = (__9.z/__10.z);
              __8.w = (__9.w/__10.w);
            int4 __11;
            ushort4 __12;
              ushort4 __13;
                ushort4 __14;
                  int4 __15 = make_int4(2, 2, 2, 2);
                  int4 __16 = make_int4(0, 0, 0, 0);
                  __14.x = (__15.x>=__16.x);
                  __14.y = (__15.y>=__16.y);
                  __14.z = (__15.z>=__16.z);
                  __14.w = (__15.w>=__16.w);
                ushort4 __17;
                  int4 __18 = make_int4(0, 0, 0, 0);
                  __17.x = (__5.x>=__18.x);
                  __17.y = (__5.y>=__18.y);
                  __17.z = (__5.z>=__18.z);
                  __17.w = (__5.w>=__18.w);
                __13.x = (__14.x&&__17.x);
                __13.y = (__14.y&&__17.y);
                __13.z = (__14.z&&__17.z);
                __13.w = (__14.w&&__17.w);
              ushort4 __19;
                ushort4 __20;
                  int4 __21 = make_int4(2, 2, 2, 2);
                  int4 __22 = make_int4(0, 0, 0, 0);
                  __20.x = (__21.x<__22.x);
                  __20.y = (__21.y<__22.y);
                  __20.z = (__21.z<__22.z);
                  __20.w = (__21.w<__22.w);
                ushort4 __23;
                  int4 __24 = make_int4(0, 0, 0, 0);
                  __23.x = (__5.x<=__24.x);
                  __23.y = (__5.y<=__24.y);
                  __23.z = (__5.z<=__24.z);
                  __23.w = (__5.w<=__24.w);
                __19.x = (__20.x&&__23.x);
                __19.y = (__20.y&&__23.y);
                __19.z = (__20.z&&__23.z);
                __19.w = (__20.w&&__23.w);
              __12.x = (__13.x||__19.x);
              __12.y = (__13.y||__19.y);
              __12.z = (__13.z||__19.z);
              __12.w = (__13.w||__19.w);
            int4 __25;
              int4 __26 = make_int4(1, 1, 1, 1);
              __25.x = (__8.x-__26.x);
              __25.y = (__8.y-__26.y);
              __25.z = (__8.z-__26.z);
              __25.w = (__8.w-__26.w);
            __11.x = (bool(__12.x)?__8.x:__25.x);
            __11.y = (bool(__12.y)?__8.y:__25.y);
            __11.z = (bool(__12.z)?__8.z:__25.z);
            __11.w = (bool(__12.w)?__8.w:__25.w);
            int __27 = ((0x000000ff << 0) & (B_local[__11.x] << 0))|((0x000000ff << 8) & (B_local[__11.y] << 8))|((0x000000ff << 16) & (B_local[__11.z] << 16))|((0x000000ff << 24) & (B_local[__11.w] << 24));
            int __28;
            int4 __29;
              int4 __30;
                int4 __31 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
                int4 __32 = make_int4(2, 2, 2, 2);
                __30.x = (__31.x%__32.x);
                __30.y = (__31.y%__32.y);
                __30.z = (__31.z%__32.z);
                __30.w = (__31.w%__32.w);
              int4 __33;
              ushort4 __34;
                ushort4 __35;
                  ushort4 __36;
                    int4 __37 = make_int4(2, 2, 2, 2);
                    int4 __38 = make_int4(0, 0, 0, 0);
                    __36.x = (__37.x>=__38.x);
                    __36.y = (__37.y>=__38.y);
                    __36.z = (__37.z>=__38.z);
                    __36.w = (__37.w>=__38.w);
                  ushort4 __39;
                    int4 __40 = make_int4(0, 0, 0, 0);
                    __39.x = (__30.x>=__40.x);
                    __39.y = (__30.y>=__40.y);
                    __39.z = (__30.z>=__40.z);
                    __39.w = (__30.w>=__40.w);
                  __35.x = (__36.x&&__39.x);
                  __35.y = (__36.y&&__39.y);
                  __35.z = (__36.z&&__39.z);
                  __35.w = (__36.w&&__39.w);
                ushort4 __41;
                  ushort4 __42;
                    int4 __43 = make_int4(2, 2, 2, 2);
                    int4 __44 = make_int4(0, 0, 0, 0);
                    __42.x = (__43.x<__44.x);
                    __42.y = (__43.y<__44.y);
                    __42.z = (__43.z<__44.z);
                    __42.w = (__43.w<__44.w);
                  ushort4 __45;
                    int4 __46 = make_int4(0, 0, 0, 0);
                    __45.x = (__30.x<=__46.x);
                    __45.y = (__30.y<=__46.y);
                    __45.z = (__30.z<=__46.z);
                    __45.w = (__30.w<=__46.w);
                  __41.x = (__42.x&&__45.x);
                  __41.y = (__42.y&&__45.y);
                  __41.z = (__42.z&&__45.z);
                  __41.w = (__42.w&&__45.w);
                __34.x = (__35.x||__41.x);
                __34.y = (__35.y||__41.y);
                __34.z = (__35.z||__41.z);
                __34.w = (__35.w||__41.w);
              int4 __47;
                int4 __48 = make_int4(2, 2, 2, 2);
                __47.x = (__30.x+__48.x);
                __47.y = (__30.y+__48.y);
                __47.z = (__30.z+__48.z);
                __47.w = (__30.w+__48.w);
              __33.x = (bool(__34.x)?__30.x:__47.x);
              __33.y = (bool(__34.y)?__30.y:__47.y);
              __33.z = (bool(__34.z)?__30.z:__47.z);
              __33.w = (bool(__34.w)?__30.w:__47.w);
              int4 __49 = make_int4(4, 4, 4, 4);
              __29.x = (__33.x*__49.x);
              __29.y = (__33.y*__49.y);
              __29.z = (__33.z*__49.z);
              __29.w = (__33.w*__49.w);
            __28=((signed char)(__29.x) << 0);
            __28=__28 & ~(0x000000ff << 8) |((signed char)(__29.y) << 8);
            __28=__28 & ~(0x000000ff << 16) |((signed char)(__29.z) << 16);
            __28=__28 & ~(0x000000ff << 24) |((signed char)(__29.w) << 24);
            __4=((((char)(__27 >> 0)) >> ((char)(__28 >> 0))) << 0);
            __4=__4 & ~(0x000000ff << 8) |((((char)(__27 >> 8)) >> ((char)(__28 >> 8))) << 8);
            __4=__4 & ~(0x000000ff << 16) |((((char)(__27 >> 16)) >> ((char)(__28 >> 16))) << 16);
            __4=__4 & ~(0x000000ff << 24) |((((char)(__27 >> 24)) >> ((char)(__28 >> 24))) << 24);
          int __50 = (int)252645135;
          __3=((((char)(__4 >> 0)) & ((char)(__50 >> 0))) << 0);
          __3=__3 & ~(0x000000ff << 8) |((((char)(__4 >> 8)) & ((char)(__50 >> 8))) << 8);
          __3=__3 & ~(0x000000ff << 16) |((((char)(__4 >> 16)) & ((char)(__50 >> 16))) << 16);
          __3=__3 & ~(0x000000ff << 24) |((((char)(__4 >> 24)) & ((char)(__50 >> 24))) << 24);
        __2.x = (int)(((char)(__3 >> 0)));
        __2.y = (int)(((char)(__3 >> 8)));
        __2.z = (int)(((char)(__3 >> 16)));
        __2.w = (int)(((char)(__3 >> 24)));
        uint2 __51 = make_uint2(__pack_half2(LUT_shared[__2.x],LUT_shared[__2.y]),__pack_half2(LUT_shared[__2.z],LUT_shared[__2.w]));
        uint2 __52 = make_uint2(__pack_half2(Scales[((((((int)blockIdx.x) * 48) + (ax0_ax1_0_fused_0_0 * 24)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))], Scales[((((((int)blockIdx.x) * 48) + (ax0_ax1_0_fused_0_0 * 24)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))]), __pack_half2(Scales[((((((int)blockIdx.x) * 48) + (ax0_ax1_0_fused_0_0 * 24)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))], Scales[((((((int)blockIdx.x) * 48) + (ax0_ax1_0_fused_0_0 * 24)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))]));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(__51.x)))->x*((half2*)(&(__52.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(__51.x)))->y*((half2*)(&(__52.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(__51.y)))->x*((half2*)(&(__52.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(__51.y)))->y*((half2*)(&(__52.y)))->y);
      *(uint2*)(B_decode_local + (ax0_0 * 4)) = __1;
    }
    *(uint4*)(B_decode_shared + (((((ax0_ax1_0_fused_0_0 * 768) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(B_decode_local + 0);
  }
__asm__ __volatile__("cp.async.commit_group;");

  #pragma unroll
  for (int ax0_ax1_fused_0_0_0_1 = 0; ax0_ax1_fused_0_0_0_1 < 1; ++ax0_ax1_fused_0_0_0_1) {
    if (((int)threadIdx.y) < 2) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + (((((((int)threadIdx.y) * 256) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 512))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(A_shared + (((((((int)threadIdx.y) * 256) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 512)))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((int)threadIdx.y) * 40960) + ((((int)threadIdx.x) >> 2) * 5120)) + ((((int)threadIdx.x) & 3) * 8)) + 32))), "n"(16)
    );
  }
    }
  }
  #pragma unroll
  for (int ax0_ax1_0_fused_0_0_1 = 0; ax0_ax1_0_fused_0_0_1 < 2; ++ax0_ax1_0_fused_0_0_1) {
    __syncthreads();

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(B_shared + ((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(B_shared + ((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.ca.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((int)blockIdx.x) * 122880) + (ax0_ax1_0_fused_0_0_1 * 61440)) + (((int)threadIdx.y) * 20480)) + ((((int)threadIdx.x) >> 2) * 2560)) + ((((int)threadIdx.x) & 3) * 4)) + 16))), "n"(4)
    );
  }
    __syncthreads();
    for (int ax0_0_1 = 0; ax0_0_1 < 2; ++ax0_0_1) {
      *(char2*)(B_local + 0) = *(char2*)(B_shared + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + (ax0_0_1 * 2)));
      uint2 __53;
        int4 __54;
        int __55;
          int __56;
            int4 __57;
              int4 __58 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __59 = make_int4(2, 2, 2, 2);
              __57.x = (__58.x%__59.x);
              __57.y = (__58.y%__59.y);
              __57.z = (__58.z%__59.z);
              __57.w = (__58.w%__59.w);
            int4 __60;
              int4 __61 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __62 = make_int4(2, 2, 2, 2);
              __60.x = (__61.x/__62.x);
              __60.y = (__61.y/__62.y);
              __60.z = (__61.z/__62.z);
              __60.w = (__61.w/__62.w);
            int4 __63;
            ushort4 __64;
              ushort4 __65;
                ushort4 __66;
                  int4 __67 = make_int4(2, 2, 2, 2);
                  int4 __68 = make_int4(0, 0, 0, 0);
                  __66.x = (__67.x>=__68.x);
                  __66.y = (__67.y>=__68.y);
                  __66.z = (__67.z>=__68.z);
                  __66.w = (__67.w>=__68.w);
                ushort4 __69;
                  int4 __70 = make_int4(0, 0, 0, 0);
                  __69.x = (__57.x>=__70.x);
                  __69.y = (__57.y>=__70.y);
                  __69.z = (__57.z>=__70.z);
                  __69.w = (__57.w>=__70.w);
                __65.x = (__66.x&&__69.x);
                __65.y = (__66.y&&__69.y);
                __65.z = (__66.z&&__69.z);
                __65.w = (__66.w&&__69.w);
              ushort4 __71;
                ushort4 __72;
                  int4 __73 = make_int4(2, 2, 2, 2);
                  int4 __74 = make_int4(0, 0, 0, 0);
                  __72.x = (__73.x<__74.x);
                  __72.y = (__73.y<__74.y);
                  __72.z = (__73.z<__74.z);
                  __72.w = (__73.w<__74.w);
                ushort4 __75;
                  int4 __76 = make_int4(0, 0, 0, 0);
                  __75.x = (__57.x<=__76.x);
                  __75.y = (__57.y<=__76.y);
                  __75.z = (__57.z<=__76.z);
                  __75.w = (__57.w<=__76.w);
                __71.x = (__72.x&&__75.x);
                __71.y = (__72.y&&__75.y);
                __71.z = (__72.z&&__75.z);
                __71.w = (__72.w&&__75.w);
              __64.x = (__65.x||__71.x);
              __64.y = (__65.y||__71.y);
              __64.z = (__65.z||__71.z);
              __64.w = (__65.w||__71.w);
            int4 __77;
              int4 __78 = make_int4(1, 1, 1, 1);
              __77.x = (__60.x-__78.x);
              __77.y = (__60.y-__78.y);
              __77.z = (__60.z-__78.z);
              __77.w = (__60.w-__78.w);
            __63.x = (bool(__64.x)?__60.x:__77.x);
            __63.y = (bool(__64.y)?__60.y:__77.y);
            __63.z = (bool(__64.z)?__60.z:__77.z);
            __63.w = (bool(__64.w)?__60.w:__77.w);
            int __79 = ((0x000000ff << 0) & (B_local[__63.x] << 0))|((0x000000ff << 8) & (B_local[__63.y] << 8))|((0x000000ff << 16) & (B_local[__63.z] << 16))|((0x000000ff << 24) & (B_local[__63.w] << 24));
            int __80;
            int4 __81;
              int4 __82;
                int4 __83 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
                int4 __84 = make_int4(2, 2, 2, 2);
                __82.x = (__83.x%__84.x);
                __82.y = (__83.y%__84.y);
                __82.z = (__83.z%__84.z);
                __82.w = (__83.w%__84.w);
              int4 __85;
              ushort4 __86;
                ushort4 __87;
                  ushort4 __88;
                    int4 __89 = make_int4(2, 2, 2, 2);
                    int4 __90 = make_int4(0, 0, 0, 0);
                    __88.x = (__89.x>=__90.x);
                    __88.y = (__89.y>=__90.y);
                    __88.z = (__89.z>=__90.z);
                    __88.w = (__89.w>=__90.w);
                  ushort4 __91;
                    int4 __92 = make_int4(0, 0, 0, 0);
                    __91.x = (__82.x>=__92.x);
                    __91.y = (__82.y>=__92.y);
                    __91.z = (__82.z>=__92.z);
                    __91.w = (__82.w>=__92.w);
                  __87.x = (__88.x&&__91.x);
                  __87.y = (__88.y&&__91.y);
                  __87.z = (__88.z&&__91.z);
                  __87.w = (__88.w&&__91.w);
                ushort4 __93;
                  ushort4 __94;
                    int4 __95 = make_int4(2, 2, 2, 2);
                    int4 __96 = make_int4(0, 0, 0, 0);
                    __94.x = (__95.x<__96.x);
                    __94.y = (__95.y<__96.y);
                    __94.z = (__95.z<__96.z);
                    __94.w = (__95.w<__96.w);
                  ushort4 __97;
                    int4 __98 = make_int4(0, 0, 0, 0);
                    __97.x = (__82.x<=__98.x);
                    __97.y = (__82.y<=__98.y);
                    __97.z = (__82.z<=__98.z);
                    __97.w = (__82.w<=__98.w);
                  __93.x = (__94.x&&__97.x);
                  __93.y = (__94.y&&__97.y);
                  __93.z = (__94.z&&__97.z);
                  __93.w = (__94.w&&__97.w);
                __86.x = (__87.x||__93.x);
                __86.y = (__87.y||__93.y);
                __86.z = (__87.z||__93.z);
                __86.w = (__87.w||__93.w);
              int4 __99;
                int4 __100 = make_int4(2, 2, 2, 2);
                __99.x = (__82.x+__100.x);
                __99.y = (__82.y+__100.y);
                __99.z = (__82.z+__100.z);
                __99.w = (__82.w+__100.w);
              __85.x = (bool(__86.x)?__82.x:__99.x);
              __85.y = (bool(__86.y)?__82.y:__99.y);
              __85.z = (bool(__86.z)?__82.z:__99.z);
              __85.w = (bool(__86.w)?__82.w:__99.w);
              int4 __101 = make_int4(4, 4, 4, 4);
              __81.x = (__85.x*__101.x);
              __81.y = (__85.y*__101.y);
              __81.z = (__85.z*__101.z);
              __81.w = (__85.w*__101.w);
            __80=((signed char)(__81.x) << 0);
            __80=__80 & ~(0x000000ff << 8) |((signed char)(__81.y) << 8);
            __80=__80 & ~(0x000000ff << 16) |((signed char)(__81.z) << 16);
            __80=__80 & ~(0x000000ff << 24) |((signed char)(__81.w) << 24);
            __56=((((char)(__79 >> 0)) >> ((char)(__80 >> 0))) << 0);
            __56=__56 & ~(0x000000ff << 8) |((((char)(__79 >> 8)) >> ((char)(__80 >> 8))) << 8);
            __56=__56 & ~(0x000000ff << 16) |((((char)(__79 >> 16)) >> ((char)(__80 >> 16))) << 16);
            __56=__56 & ~(0x000000ff << 24) |((((char)(__79 >> 24)) >> ((char)(__80 >> 24))) << 24);
          int __102 = (int)252645135;
          __55=((((char)(__56 >> 0)) & ((char)(__102 >> 0))) << 0);
          __55=__55 & ~(0x000000ff << 8) |((((char)(__56 >> 8)) & ((char)(__102 >> 8))) << 8);
          __55=__55 & ~(0x000000ff << 16) |((((char)(__56 >> 16)) & ((char)(__102 >> 16))) << 16);
          __55=__55 & ~(0x000000ff << 24) |((((char)(__56 >> 24)) & ((char)(__102 >> 24))) << 24);
        __54.x = (int)(((char)(__55 >> 0)));
        __54.y = (int)(((char)(__55 >> 8)));
        __54.z = (int)(((char)(__55 >> 16)));
        __54.w = (int)(((char)(__55 >> 24)));
        uint2 __103 = make_uint2(__pack_half2(LUT_shared[__54.x],LUT_shared[__54.y]),__pack_half2(LUT_shared[__54.z],LUT_shared[__54.w]));
        uint2 __104 = make_uint2(__pack_half2(Scales[((((((int)blockIdx.x) * 48) + (ax0_ax1_0_fused_0_0_1 * 24)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))], Scales[((((((int)blockIdx.x) * 48) + (ax0_ax1_0_fused_0_0_1 * 24)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))]), __pack_half2(Scales[((((((int)blockIdx.x) * 48) + (ax0_ax1_0_fused_0_0_1 * 24)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))], Scales[((((((int)blockIdx.x) * 48) + (ax0_ax1_0_fused_0_0_1 * 24)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))]));
        ((half2*)(&(__53.x)))->x = (((half2*)(&(__103.x)))->x*((half2*)(&(__104.x)))->x);
        ((half2*)(&(__53.x)))->y = (((half2*)(&(__103.x)))->y*((half2*)(&(__104.x)))->y);
        ((half2*)(&(__53.y)))->x = (((half2*)(&(__103.y)))->x*((half2*)(&(__104.y)))->x);
        ((half2*)(&(__53.y)))->y = (((half2*)(&(__103.y)))->y*((half2*)(&(__104.y)))->y);
      *(uint2*)(B_decode_local + (ax0_0_1 * 4)) = __53;
    }
    *(uint4*)(B_decode_shared + ((((((ax0_ax1_0_fused_0_0_1 * 768) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 1536)) = *(uint4*)(B_decode_local + 0);
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 1;");

  __syncthreads();
  call_cutlass_mma_prologue(C_cutlass_warp_mma, (&(A_shared[0])), (&(B_decode_shared[0])), 32, 32);
  call_cutlass_mma_body(C_cutlass_warp_mma);
  for (int k_0 = 0; k_0 < 158; ++k_0) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_0_0_0_2 = 0; ax0_ax1_fused_0_0_0_2 < 1; ++ax0_ax1_fused_0_0_0_2) {
      if (((int)threadIdx.y) < 2) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + ((((((k_0 & 1) * 512) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(A_shared + ((((((k_0 & 1) * 512) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + (((((((int)threadIdx.y) * 40960) + ((((int)threadIdx.x) >> 2) * 5120)) + (k_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 64))), "n"(16)
    );
  }
      }
    }
    #pragma unroll
    for (int ax0_ax1_0_fused_0_0_2 = 0; ax0_ax1_0_fused_0_0_2 < 2; ++ax0_ax1_0_fused_0_0_2) {
      __syncthreads();

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(B_shared + ((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(B_shared + ((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.ca.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + (((((((((int)blockIdx.x) * 122880) + (ax0_ax1_0_fused_0_0_2 * 61440)) + (((int)threadIdx.y) * 20480)) + ((((int)threadIdx.x) >> 2) * 2560)) + (k_0 * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 32))), "n"(4)
    );
  }
      __syncthreads();
      for (int ax0_0_2 = 0; ax0_0_2 < 2; ++ax0_0_2) {
        *(char2*)(B_local_1 + 0) = *(char2*)(B_shared + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + (ax0_0_2 * 2)));
        uint2 __105;
          int4 __106;
          int __107;
            int __108;
              int4 __109;
                int4 __110 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
                int4 __111 = make_int4(2, 2, 2, 2);
                __109.x = (__110.x%__111.x);
                __109.y = (__110.y%__111.y);
                __109.z = (__110.z%__111.z);
                __109.w = (__110.w%__111.w);
              int4 __112;
                int4 __113 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
                int4 __114 = make_int4(2, 2, 2, 2);
                __112.x = (__113.x/__114.x);
                __112.y = (__113.y/__114.y);
                __112.z = (__113.z/__114.z);
                __112.w = (__113.w/__114.w);
              int4 __115;
              ushort4 __116;
                ushort4 __117;
                  ushort4 __118;
                    int4 __119 = make_int4(2, 2, 2, 2);
                    int4 __120 = make_int4(0, 0, 0, 0);
                    __118.x = (__119.x>=__120.x);
                    __118.y = (__119.y>=__120.y);
                    __118.z = (__119.z>=__120.z);
                    __118.w = (__119.w>=__120.w);
                  ushort4 __121;
                    int4 __122 = make_int4(0, 0, 0, 0);
                    __121.x = (__109.x>=__122.x);
                    __121.y = (__109.y>=__122.y);
                    __121.z = (__109.z>=__122.z);
                    __121.w = (__109.w>=__122.w);
                  __117.x = (__118.x&&__121.x);
                  __117.y = (__118.y&&__121.y);
                  __117.z = (__118.z&&__121.z);
                  __117.w = (__118.w&&__121.w);
                ushort4 __123;
                  ushort4 __124;
                    int4 __125 = make_int4(2, 2, 2, 2);
                    int4 __126 = make_int4(0, 0, 0, 0);
                    __124.x = (__125.x<__126.x);
                    __124.y = (__125.y<__126.y);
                    __124.z = (__125.z<__126.z);
                    __124.w = (__125.w<__126.w);
                  ushort4 __127;
                    int4 __128 = make_int4(0, 0, 0, 0);
                    __127.x = (__109.x<=__128.x);
                    __127.y = (__109.y<=__128.y);
                    __127.z = (__109.z<=__128.z);
                    __127.w = (__109.w<=__128.w);
                  __123.x = (__124.x&&__127.x);
                  __123.y = (__124.y&&__127.y);
                  __123.z = (__124.z&&__127.z);
                  __123.w = (__124.w&&__127.w);
                __116.x = (__117.x||__123.x);
                __116.y = (__117.y||__123.y);
                __116.z = (__117.z||__123.z);
                __116.w = (__117.w||__123.w);
              int4 __129;
                int4 __130 = make_int4(1, 1, 1, 1);
                __129.x = (__112.x-__130.x);
                __129.y = (__112.y-__130.y);
                __129.z = (__112.z-__130.z);
                __129.w = (__112.w-__130.w);
              __115.x = (bool(__116.x)?__112.x:__129.x);
              __115.y = (bool(__116.y)?__112.y:__129.y);
              __115.z = (bool(__116.z)?__112.z:__129.z);
              __115.w = (bool(__116.w)?__112.w:__129.w);
              int __131 = ((0x000000ff << 0) & (B_local_1[__115.x] << 0))|((0x000000ff << 8) & (B_local_1[__115.y] << 8))|((0x000000ff << 16) & (B_local_1[__115.z] << 16))|((0x000000ff << 24) & (B_local_1[__115.w] << 24));
              int __132;
              int4 __133;
                int4 __134;
                  int4 __135 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
                  int4 __136 = make_int4(2, 2, 2, 2);
                  __134.x = (__135.x%__136.x);
                  __134.y = (__135.y%__136.y);
                  __134.z = (__135.z%__136.z);
                  __134.w = (__135.w%__136.w);
                int4 __137;
                ushort4 __138;
                  ushort4 __139;
                    ushort4 __140;
                      int4 __141 = make_int4(2, 2, 2, 2);
                      int4 __142 = make_int4(0, 0, 0, 0);
                      __140.x = (__141.x>=__142.x);
                      __140.y = (__141.y>=__142.y);
                      __140.z = (__141.z>=__142.z);
                      __140.w = (__141.w>=__142.w);
                    ushort4 __143;
                      int4 __144 = make_int4(0, 0, 0, 0);
                      __143.x = (__134.x>=__144.x);
                      __143.y = (__134.y>=__144.y);
                      __143.z = (__134.z>=__144.z);
                      __143.w = (__134.w>=__144.w);
                    __139.x = (__140.x&&__143.x);
                    __139.y = (__140.y&&__143.y);
                    __139.z = (__140.z&&__143.z);
                    __139.w = (__140.w&&__143.w);
                  ushort4 __145;
                    ushort4 __146;
                      int4 __147 = make_int4(2, 2, 2, 2);
                      int4 __148 = make_int4(0, 0, 0, 0);
                      __146.x = (__147.x<__148.x);
                      __146.y = (__147.y<__148.y);
                      __146.z = (__147.z<__148.z);
                      __146.w = (__147.w<__148.w);
                    ushort4 __149;
                      int4 __150 = make_int4(0, 0, 0, 0);
                      __149.x = (__134.x<=__150.x);
                      __149.y = (__134.y<=__150.y);
                      __149.z = (__134.z<=__150.z);
                      __149.w = (__134.w<=__150.w);
                    __145.x = (__146.x&&__149.x);
                    __145.y = (__146.y&&__149.y);
                    __145.z = (__146.z&&__149.z);
                    __145.w = (__146.w&&__149.w);
                  __138.x = (__139.x||__145.x);
                  __138.y = (__139.y||__145.y);
                  __138.z = (__139.z||__145.z);
                  __138.w = (__139.w||__145.w);
                int4 __151;
                  int4 __152 = make_int4(2, 2, 2, 2);
                  __151.x = (__134.x+__152.x);
                  __151.y = (__134.y+__152.y);
                  __151.z = (__134.z+__152.z);
                  __151.w = (__134.w+__152.w);
                __137.x = (bool(__138.x)?__134.x:__151.x);
                __137.y = (bool(__138.y)?__134.y:__151.y);
                __137.z = (bool(__138.z)?__134.z:__151.z);
                __137.w = (bool(__138.w)?__134.w:__151.w);
                int4 __153 = make_int4(4, 4, 4, 4);
                __133.x = (__137.x*__153.x);
                __133.y = (__137.y*__153.y);
                __133.z = (__137.z*__153.z);
                __133.w = (__137.w*__153.w);
              __132=((signed char)(__133.x) << 0);
              __132=__132 & ~(0x000000ff << 8) |((signed char)(__133.y) << 8);
              __132=__132 & ~(0x000000ff << 16) |((signed char)(__133.z) << 16);
              __132=__132 & ~(0x000000ff << 24) |((signed char)(__133.w) << 24);
              __108=((((char)(__131 >> 0)) >> ((char)(__132 >> 0))) << 0);
              __108=__108 & ~(0x000000ff << 8) |((((char)(__131 >> 8)) >> ((char)(__132 >> 8))) << 8);
              __108=__108 & ~(0x000000ff << 16) |((((char)(__131 >> 16)) >> ((char)(__132 >> 16))) << 16);
              __108=__108 & ~(0x000000ff << 24) |((((char)(__131 >> 24)) >> ((char)(__132 >> 24))) << 24);
            int __154 = (int)252645135;
            __107=((((char)(__108 >> 0)) & ((char)(__154 >> 0))) << 0);
            __107=__107 & ~(0x000000ff << 8) |((((char)(__108 >> 8)) & ((char)(__154 >> 8))) << 8);
            __107=__107 & ~(0x000000ff << 16) |((((char)(__108 >> 16)) & ((char)(__154 >> 16))) << 16);
            __107=__107 & ~(0x000000ff << 24) |((((char)(__108 >> 24)) & ((char)(__154 >> 24))) << 24);
          __106.x = (int)(((char)(__107 >> 0)));
          __106.y = (int)(((char)(__107 >> 8)));
          __106.z = (int)(((char)(__107 >> 16)));
          __106.w = (int)(((char)(__107 >> 24)));
          uint2 __155 = make_uint2(__pack_half2(LUT_shared[__106.x],LUT_shared[__106.y]),__pack_half2(LUT_shared[__106.z],LUT_shared[__106.w]));
          int4 __156 = make_int4(((((((((((k_0 * 32) + ((((int)threadIdx.x) & 3) * 8)) + (ax0_0_2 * 4)) + 64) >> 7) * 15360) + (((int)blockIdx.x) * 48)) + (ax0_ax1_0_fused_0_0_2 * 24)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)), ((((((((((k_0 * 32) + ((((int)threadIdx.x) & 3) * 8)) + (ax0_0_2 * 4)) + 64) >> 7) * 15360) + (((int)blockIdx.x) * 48)) + (ax0_ax1_0_fused_0_0_2 * 24)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)), ((((((((((k_0 * 32) + ((((int)threadIdx.x) & 3) * 8)) + (ax0_0_2 * 4)) + 64) >> 7) * 15360) + (((int)blockIdx.x) * 48)) + (ax0_ax1_0_fused_0_0_2 * 24)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)), ((((((((((k_0 * 32) + ((((int)threadIdx.x) & 3) * 8)) + (ax0_0_2 * 4)) + 64) >> 7) * 15360) + (((int)blockIdx.x) * 48)) + (ax0_ax1_0_fused_0_0_2 * 24)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)));
          uint2 __157 = make_uint2(__pack_half2(Scales[__156.x],Scales[__156.y]),__pack_half2(Scales[__156.z],Scales[__156.w]));
          ((half2*)(&(__105.x)))->x = (((half2*)(&(__155.x)))->x*((half2*)(&(__157.x)))->x);
          ((half2*)(&(__105.x)))->y = (((half2*)(&(__155.x)))->y*((half2*)(&(__157.x)))->y);
          ((half2*)(&(__105.y)))->x = (((half2*)(&(__155.y)))->x*((half2*)(&(__157.y)))->x);
          ((half2*)(&(__105.y)))->y = (((half2*)(&(__155.y)))->y*((half2*)(&(__157.y)))->y);
        *(uint2*)(B_decode_local_1 + (ax0_0_2 * 4)) = __105;
      }
      *(uint4*)(B_decode_shared + (((((((k_0 & 1) * 1536) + (ax0_ax1_0_fused_0_0_2 * 768)) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(B_decode_local_1 + 0);
    }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 1;");

    __syncthreads();
    call_cutlass_mma_prologue(C_cutlass_warp_mma, (&(A_shared[(((k_0 + 1) & 1) * 512)])), (&(B_decode_shared[(((k_0 + 1) & 1) * 1536)])), 32, 32);
    call_cutlass_mma_epilogue(C_cutlass_warp_mma);
    call_cutlass_mma_body(C_cutlass_warp_mma);
  }
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  call_cutlass_mma_prologue(C_cutlass_warp_mma, (&(A_shared[512])), (&(B_decode_shared[1536])), 32, 32);
  call_cutlass_mma_epilogue(C_cutlass_warp_mma);
  call_cutlass_mma_body(C_cutlass_warp_mma);
  call_cutlass_mma_epilogue(C_cutlass_warp_mma);
  #pragma unroll
  for (int ax1_0 = 0; ax1_0 < 4; ++ax1_0) {
    *(uint1*)(C + (((((((ax1_0 & 1) * 122880) + ((((int)threadIdx.x) >> 2) * 15360)) + (((int)blockIdx.x) * 48)) + (((int)threadIdx.y) * 16)) + ((ax1_0 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(C_cutlass_warp_mma + (ax1_0 * 2));
  }
}



__global__ void __launch_bounds__(32) cutlass_kernel_fp16_nf4_fp16_m16n5120k5120_nt_8x32x32_8x32x16_(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ C) {
            int8_t* B = ((int8_t *)QB);
	 half* LUT = (half *)((int8_t *)QB + 13107200);
	 half* Scales = (half *)((int8_t *)QB + 13107200 + 32);                 
            // const dim3 GridDim(320, 1, 1);
            // const dim3 BlockDim(32, 1, 1);
            // cutlass_kernel_fp16_nf4_fp16_m16n5120k5120_nt_8x32x32_8x32x16_<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
  
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, half> C_wmma_accumulator[1];
  __shared__ half A_shared[320];
  __shared__ half B_decode_shared[1280];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, half, nvcuda::wmma::row_major> A_shared_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, half, nvcuda::wmma::col_major> B_decode_shared_wmma_matrix_b[1];
  nvcuda::wmma::fill_fragment(C_wmma_accumulator[0], __float2half_rn(0.000000e+00f));
  for (int k_outer = 0; k_outer < 160; ++k_outer) {
    __syncthreads();
    *(uint4*)(A_shared + (((((int)threadIdx.x) >> 2) * 40) + ((((int)threadIdx.x) & 3) * 8))) = *(uint4*)(A + (((((((int)blockIdx.x) / 160) * 40960) + ((((int)threadIdx.x) >> 2) * 5120)) + (k_outer * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    uint2 __1;
      int4 __2;
      int __3;
        int __4;
          int4 __5;
            int4 __6 = make_int4((((((((int)blockIdx.x) % 160) * 81920) + ((((int)threadIdx.x) >> 3) * 2560)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)), (((((((int)blockIdx.x) % 160) * 81920) + ((((int)threadIdx.x) >> 3) * 2560)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)), (((((((int)blockIdx.x) % 160) * 81920) + ((((int)threadIdx.x) >> 3) * 2560)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)), (((((((int)blockIdx.x) % 160) * 81920) + ((((int)threadIdx.x) >> 3) * 2560)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)));
            int4 __7;
              int4 __8 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __9 = make_int4(2, 2, 2, 2);
              __7.x = (__8.x%__9.x);
              __7.y = (__8.y%__9.y);
              __7.z = (__8.z%__9.z);
              __7.w = (__8.w%__9.w);
            int4 __10;
              int4 __11 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __12 = make_int4(2, 2, 2, 2);
              __10.x = (__11.x/__12.x);
              __10.y = (__11.y/__12.y);
              __10.z = (__11.z/__12.z);
              __10.w = (__11.w/__12.w);
            int4 __13;
            ushort4 __14;
              ushort4 __15;
                ushort4 __16;
                  int4 __17 = make_int4(2, 2, 2, 2);
                  int4 __18 = make_int4(0, 0, 0, 0);
                  __16.x = (__17.x>=__18.x);
                  __16.y = (__17.y>=__18.y);
                  __16.z = (__17.z>=__18.z);
                  __16.w = (__17.w>=__18.w);
                ushort4 __19;
                  int4 __20 = make_int4(0, 0, 0, 0);
                  __19.x = (__7.x>=__20.x);
                  __19.y = (__7.y>=__20.y);
                  __19.z = (__7.z>=__20.z);
                  __19.w = (__7.w>=__20.w);
                __15.x = (__16.x&&__19.x);
                __15.y = (__16.y&&__19.y);
                __15.z = (__16.z&&__19.z);
                __15.w = (__16.w&&__19.w);
              ushort4 __21;
                ushort4 __22;
                  int4 __23 = make_int4(2, 2, 2, 2);
                  int4 __24 = make_int4(0, 0, 0, 0);
                  __22.x = (__23.x<__24.x);
                  __22.y = (__23.y<__24.y);
                  __22.z = (__23.z<__24.z);
                  __22.w = (__23.w<__24.w);
                ushort4 __25;
                  int4 __26 = make_int4(0, 0, 0, 0);
                  __25.x = (__7.x<=__26.x);
                  __25.y = (__7.y<=__26.y);
                  __25.z = (__7.z<=__26.z);
                  __25.w = (__7.w<=__26.w);
                __21.x = (__22.x&&__25.x);
                __21.y = (__22.y&&__25.y);
                __21.z = (__22.z&&__25.z);
                __21.w = (__22.w&&__25.w);
              __14.x = (__15.x||__21.x);
              __14.y = (__15.y||__21.y);
              __14.z = (__15.z||__21.z);
              __14.w = (__15.w||__21.w);
            int4 __27;
              int4 __28 = make_int4(1, 1, 1, 1);
              __27.x = (__10.x-__28.x);
              __27.y = (__10.y-__28.y);
              __27.z = (__10.z-__28.z);
              __27.w = (__10.w-__28.w);
            __13.x = (bool(__14.x)?__10.x:__27.x);
            __13.y = (bool(__14.y)?__10.y:__27.y);
            __13.z = (bool(__14.z)?__10.z:__27.z);
            __13.w = (bool(__14.w)?__10.w:__27.w);
            __5.x = (__6.x+__13.x);
            __5.y = (__6.y+__13.y);
            __5.z = (__6.z+__13.z);
            __5.w = (__6.w+__13.w);
          int __29 = ((0x000000ff << 0) & (B[__5.x] << 0))|((0x000000ff << 8) & (B[__5.y] << 8))|((0x000000ff << 16) & (B[__5.z] << 16))|((0x000000ff << 24) & (B[__5.w] << 24));
          int __30;
          int4 __31;
            int4 __32;
              int4 __33 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __34 = make_int4(2, 2, 2, 2);
              __32.x = (__33.x%__34.x);
              __32.y = (__33.y%__34.y);
              __32.z = (__33.z%__34.z);
              __32.w = (__33.w%__34.w);
            int4 __35;
            ushort4 __36;
              ushort4 __37;
                ushort4 __38;
                  int4 __39 = make_int4(2, 2, 2, 2);
                  int4 __40 = make_int4(0, 0, 0, 0);
                  __38.x = (__39.x>=__40.x);
                  __38.y = (__39.y>=__40.y);
                  __38.z = (__39.z>=__40.z);
                  __38.w = (__39.w>=__40.w);
                ushort4 __41;
                  int4 __42 = make_int4(0, 0, 0, 0);
                  __41.x = (__32.x>=__42.x);
                  __41.y = (__32.y>=__42.y);
                  __41.z = (__32.z>=__42.z);
                  __41.w = (__32.w>=__42.w);
                __37.x = (__38.x&&__41.x);
                __37.y = (__38.y&&__41.y);
                __37.z = (__38.z&&__41.z);
                __37.w = (__38.w&&__41.w);
              ushort4 __43;
                ushort4 __44;
                  int4 __45 = make_int4(2, 2, 2, 2);
                  int4 __46 = make_int4(0, 0, 0, 0);
                  __44.x = (__45.x<__46.x);
                  __44.y = (__45.y<__46.y);
                  __44.z = (__45.z<__46.z);
                  __44.w = (__45.w<__46.w);
                ushort4 __47;
                  int4 __48 = make_int4(0, 0, 0, 0);
                  __47.x = (__32.x<=__48.x);
                  __47.y = (__32.y<=__48.y);
                  __47.z = (__32.z<=__48.z);
                  __47.w = (__32.w<=__48.w);
                __43.x = (__44.x&&__47.x);
                __43.y = (__44.y&&__47.y);
                __43.z = (__44.z&&__47.z);
                __43.w = (__44.w&&__47.w);
              __36.x = (__37.x||__43.x);
              __36.y = (__37.y||__43.y);
              __36.z = (__37.z||__43.z);
              __36.w = (__37.w||__43.w);
            int4 __49;
              int4 __50 = make_int4(2, 2, 2, 2);
              __49.x = (__32.x+__50.x);
              __49.y = (__32.y+__50.y);
              __49.z = (__32.z+__50.z);
              __49.w = (__32.w+__50.w);
            __35.x = (bool(__36.x)?__32.x:__49.x);
            __35.y = (bool(__36.y)?__32.y:__49.y);
            __35.z = (bool(__36.z)?__32.z:__49.z);
            __35.w = (bool(__36.w)?__32.w:__49.w);
            int4 __51 = make_int4(4, 4, 4, 4);
            __31.x = (__35.x*__51.x);
            __31.y = (__35.y*__51.y);
            __31.z = (__35.z*__51.z);
            __31.w = (__35.w*__51.w);
          __30=((signed char)(__31.x) << 0);
          __30=__30 & ~(0x000000ff << 8) |((signed char)(__31.y) << 8);
          __30=__30 & ~(0x000000ff << 16) |((signed char)(__31.z) << 16);
          __30=__30 & ~(0x000000ff << 24) |((signed char)(__31.w) << 24);
          __4=((((char)(__29 >> 0)) >> ((char)(__30 >> 0))) << 0);
          __4=__4 & ~(0x000000ff << 8) |((((char)(__29 >> 8)) >> ((char)(__30 >> 8))) << 8);
          __4=__4 & ~(0x000000ff << 16) |((((char)(__29 >> 16)) >> ((char)(__30 >> 16))) << 16);
          __4=__4 & ~(0x000000ff << 24) |((((char)(__29 >> 24)) >> ((char)(__30 >> 24))) << 24);
        int __52 = (int)252645135;
        __3=((((char)(__4 >> 0)) & ((char)(__52 >> 0))) << 0);
        __3=__3 & ~(0x000000ff << 8) |((((char)(__4 >> 8)) & ((char)(__52 >> 8))) << 8);
        __3=__3 & ~(0x000000ff << 16) |((((char)(__4 >> 16)) & ((char)(__52 >> 16))) << 16);
        __3=__3 & ~(0x000000ff << 24) |((((char)(__4 >> 24)) & ((char)(__52 >> 24))) << 24);
      __2.x = (int)(((char)(__3 >> 0)));
      __2.y = (int)(((char)(__3 >> 8)));
      __2.z = (int)(((char)(__3 >> 16)));
      __2.w = (int)(((char)(__3 >> 24)));
      uint2 __53 = make_uint2(__pack_half2(LUT[__2.x],LUT[__2.y]),__pack_half2(LUT[__2.z],LUT[__2.w]));
      uint2 __54 = make_uint2(__pack_half2(Scales[((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3))], Scales[((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3))]), __pack_half2(Scales[((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3))], Scales[((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3))]));
      ((half2*)(&(__1.x)))->x = (((half2*)(&(__53.x)))->x*((half2*)(&(__54.x)))->x);
      ((half2*)(&(__1.x)))->y = (((half2*)(&(__53.x)))->y*((half2*)(&(__54.x)))->y);
      ((half2*)(&(__1.y)))->x = (((half2*)(&(__53.y)))->x*((half2*)(&(__54.y)))->x);
      ((half2*)(&(__1.y)))->y = (((half2*)(&(__53.y)))->y*((half2*)(&(__54.y)))->y);
    *(uint2*)(B_decode_shared + (((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 4))) = __1;
    uint2 __55;
      int4 __56;
      int __57;
        int __58;
          int4 __59;
            int4 __60 = make_int4(((((((((int)blockIdx.x) % 160) * 81920) + ((((int)threadIdx.x) >> 3) * 2560)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 10240), ((((((((int)blockIdx.x) % 160) * 81920) + ((((int)threadIdx.x) >> 3) * 2560)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 10240), ((((((((int)blockIdx.x) % 160) * 81920) + ((((int)threadIdx.x) >> 3) * 2560)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 10240), ((((((((int)blockIdx.x) % 160) * 81920) + ((((int)threadIdx.x) >> 3) * 2560)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 10240));
            int4 __61;
              int4 __62 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __63 = make_int4(2, 2, 2, 2);
              __61.x = (__62.x%__63.x);
              __61.y = (__62.y%__63.y);
              __61.z = (__62.z%__63.z);
              __61.w = (__62.w%__63.w);
            int4 __64;
              int4 __65 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __66 = make_int4(2, 2, 2, 2);
              __64.x = (__65.x/__66.x);
              __64.y = (__65.y/__66.y);
              __64.z = (__65.z/__66.z);
              __64.w = (__65.w/__66.w);
            int4 __67;
            ushort4 __68;
              ushort4 __69;
                ushort4 __70;
                  int4 __71 = make_int4(2, 2, 2, 2);
                  int4 __72 = make_int4(0, 0, 0, 0);
                  __70.x = (__71.x>=__72.x);
                  __70.y = (__71.y>=__72.y);
                  __70.z = (__71.z>=__72.z);
                  __70.w = (__71.w>=__72.w);
                ushort4 __73;
                  int4 __74 = make_int4(0, 0, 0, 0);
                  __73.x = (__61.x>=__74.x);
                  __73.y = (__61.y>=__74.y);
                  __73.z = (__61.z>=__74.z);
                  __73.w = (__61.w>=__74.w);
                __69.x = (__70.x&&__73.x);
                __69.y = (__70.y&&__73.y);
                __69.z = (__70.z&&__73.z);
                __69.w = (__70.w&&__73.w);
              ushort4 __75;
                ushort4 __76;
                  int4 __77 = make_int4(2, 2, 2, 2);
                  int4 __78 = make_int4(0, 0, 0, 0);
                  __76.x = (__77.x<__78.x);
                  __76.y = (__77.y<__78.y);
                  __76.z = (__77.z<__78.z);
                  __76.w = (__77.w<__78.w);
                ushort4 __79;
                  int4 __80 = make_int4(0, 0, 0, 0);
                  __79.x = (__61.x<=__80.x);
                  __79.y = (__61.y<=__80.y);
                  __79.z = (__61.z<=__80.z);
                  __79.w = (__61.w<=__80.w);
                __75.x = (__76.x&&__79.x);
                __75.y = (__76.y&&__79.y);
                __75.z = (__76.z&&__79.z);
                __75.w = (__76.w&&__79.w);
              __68.x = (__69.x||__75.x);
              __68.y = (__69.y||__75.y);
              __68.z = (__69.z||__75.z);
              __68.w = (__69.w||__75.w);
            int4 __81;
              int4 __82 = make_int4(1, 1, 1, 1);
              __81.x = (__64.x-__82.x);
              __81.y = (__64.y-__82.y);
              __81.z = (__64.z-__82.z);
              __81.w = (__64.w-__82.w);
            __67.x = (bool(__68.x)?__64.x:__81.x);
            __67.y = (bool(__68.y)?__64.y:__81.y);
            __67.z = (bool(__68.z)?__64.z:__81.z);
            __67.w = (bool(__68.w)?__64.w:__81.w);
            __59.x = (__60.x+__67.x);
            __59.y = (__60.y+__67.y);
            __59.z = (__60.z+__67.z);
            __59.w = (__60.w+__67.w);
          int __83 = ((0x000000ff << 0) & (B[__59.x] << 0))|((0x000000ff << 8) & (B[__59.y] << 8))|((0x000000ff << 16) & (B[__59.z] << 16))|((0x000000ff << 24) & (B[__59.w] << 24));
          int __84;
          int4 __85;
            int4 __86;
              int4 __87 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __88 = make_int4(2, 2, 2, 2);
              __86.x = (__87.x%__88.x);
              __86.y = (__87.y%__88.y);
              __86.z = (__87.z%__88.z);
              __86.w = (__87.w%__88.w);
            int4 __89;
            ushort4 __90;
              ushort4 __91;
                ushort4 __92;
                  int4 __93 = make_int4(2, 2, 2, 2);
                  int4 __94 = make_int4(0, 0, 0, 0);
                  __92.x = (__93.x>=__94.x);
                  __92.y = (__93.y>=__94.y);
                  __92.z = (__93.z>=__94.z);
                  __92.w = (__93.w>=__94.w);
                ushort4 __95;
                  int4 __96 = make_int4(0, 0, 0, 0);
                  __95.x = (__86.x>=__96.x);
                  __95.y = (__86.y>=__96.y);
                  __95.z = (__86.z>=__96.z);
                  __95.w = (__86.w>=__96.w);
                __91.x = (__92.x&&__95.x);
                __91.y = (__92.y&&__95.y);
                __91.z = (__92.z&&__95.z);
                __91.w = (__92.w&&__95.w);
              ushort4 __97;
                ushort4 __98;
                  int4 __99 = make_int4(2, 2, 2, 2);
                  int4 __100 = make_int4(0, 0, 0, 0);
                  __98.x = (__99.x<__100.x);
                  __98.y = (__99.y<__100.y);
                  __98.z = (__99.z<__100.z);
                  __98.w = (__99.w<__100.w);
                ushort4 __101;
                  int4 __102 = make_int4(0, 0, 0, 0);
                  __101.x = (__86.x<=__102.x);
                  __101.y = (__86.y<=__102.y);
                  __101.z = (__86.z<=__102.z);
                  __101.w = (__86.w<=__102.w);
                __97.x = (__98.x&&__101.x);
                __97.y = (__98.y&&__101.y);
                __97.z = (__98.z&&__101.z);
                __97.w = (__98.w&&__101.w);
              __90.x = (__91.x||__97.x);
              __90.y = (__91.y||__97.y);
              __90.z = (__91.z||__97.z);
              __90.w = (__91.w||__97.w);
            int4 __103;
              int4 __104 = make_int4(2, 2, 2, 2);
              __103.x = (__86.x+__104.x);
              __103.y = (__86.y+__104.y);
              __103.z = (__86.z+__104.z);
              __103.w = (__86.w+__104.w);
            __89.x = (bool(__90.x)?__86.x:__103.x);
            __89.y = (bool(__90.y)?__86.y:__103.y);
            __89.z = (bool(__90.z)?__86.z:__103.z);
            __89.w = (bool(__90.w)?__86.w:__103.w);
            int4 __105 = make_int4(4, 4, 4, 4);
            __85.x = (__89.x*__105.x);
            __85.y = (__89.y*__105.y);
            __85.z = (__89.z*__105.z);
            __85.w = (__89.w*__105.w);
          __84=((signed char)(__85.x) << 0);
          __84=__84 & ~(0x000000ff << 8) |((signed char)(__85.y) << 8);
          __84=__84 & ~(0x000000ff << 16) |((signed char)(__85.z) << 16);
          __84=__84 & ~(0x000000ff << 24) |((signed char)(__85.w) << 24);
          __58=((((char)(__83 >> 0)) >> ((char)(__84 >> 0))) << 0);
          __58=__58 & ~(0x000000ff << 8) |((((char)(__83 >> 8)) >> ((char)(__84 >> 8))) << 8);
          __58=__58 & ~(0x000000ff << 16) |((((char)(__83 >> 16)) >> ((char)(__84 >> 16))) << 16);
          __58=__58 & ~(0x000000ff << 24) |((((char)(__83 >> 24)) >> ((char)(__84 >> 24))) << 24);
        int __106 = (int)252645135;
        __57=((((char)(__58 >> 0)) & ((char)(__106 >> 0))) << 0);
        __57=__57 & ~(0x000000ff << 8) |((((char)(__58 >> 8)) & ((char)(__106 >> 8))) << 8);
        __57=__57 & ~(0x000000ff << 16) |((((char)(__58 >> 16)) & ((char)(__106 >> 16))) << 16);
        __57=__57 & ~(0x000000ff << 24) |((((char)(__58 >> 24)) & ((char)(__106 >> 24))) << 24);
      __56.x = (int)(((char)(__57 >> 0)));
      __56.y = (int)(((char)(__57 >> 8)));
      __56.z = (int)(((char)(__57 >> 16)));
      __56.w = (int)(((char)(__57 >> 24)));
      uint2 __107 = make_uint2(__pack_half2(LUT[__56.x],LUT[__56.y]),__pack_half2(LUT[__56.z],LUT[__56.w]));
      uint2 __108 = make_uint2(__pack_half2(Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 4)], Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 4)]), __pack_half2(Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 4)], Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 4)]));
      ((half2*)(&(__55.x)))->x = (((half2*)(&(__107.x)))->x*((half2*)(&(__108.x)))->x);
      ((half2*)(&(__55.x)))->y = (((half2*)(&(__107.x)))->y*((half2*)(&(__108.x)))->y);
      ((half2*)(&(__55.y)))->x = (((half2*)(&(__107.y)))->x*((half2*)(&(__108.y)))->x);
      ((half2*)(&(__55.y)))->y = (((half2*)(&(__107.y)))->y*((half2*)(&(__108.y)))->y);
    *(uint2*)(B_decode_shared + ((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 4)) + 160)) = __55;
    uint2 __109;
      int4 __110;
      int __111;
        int __112;
          int4 __113;
            int4 __114 = make_int4(((((((((int)blockIdx.x) % 160) * 81920) + ((((int)threadIdx.x) >> 3) * 2560)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 20480), ((((((((int)blockIdx.x) % 160) * 81920) + ((((int)threadIdx.x) >> 3) * 2560)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 20480), ((((((((int)blockIdx.x) % 160) * 81920) + ((((int)threadIdx.x) >> 3) * 2560)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 20480), ((((((((int)blockIdx.x) % 160) * 81920) + ((((int)threadIdx.x) >> 3) * 2560)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 20480));
            int4 __115;
              int4 __116 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __117 = make_int4(2, 2, 2, 2);
              __115.x = (__116.x%__117.x);
              __115.y = (__116.y%__117.y);
              __115.z = (__116.z%__117.z);
              __115.w = (__116.w%__117.w);
            int4 __118;
              int4 __119 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __120 = make_int4(2, 2, 2, 2);
              __118.x = (__119.x/__120.x);
              __118.y = (__119.y/__120.y);
              __118.z = (__119.z/__120.z);
              __118.w = (__119.w/__120.w);
            int4 __121;
            ushort4 __122;
              ushort4 __123;
                ushort4 __124;
                  int4 __125 = make_int4(2, 2, 2, 2);
                  int4 __126 = make_int4(0, 0, 0, 0);
                  __124.x = (__125.x>=__126.x);
                  __124.y = (__125.y>=__126.y);
                  __124.z = (__125.z>=__126.z);
                  __124.w = (__125.w>=__126.w);
                ushort4 __127;
                  int4 __128 = make_int4(0, 0, 0, 0);
                  __127.x = (__115.x>=__128.x);
                  __127.y = (__115.y>=__128.y);
                  __127.z = (__115.z>=__128.z);
                  __127.w = (__115.w>=__128.w);
                __123.x = (__124.x&&__127.x);
                __123.y = (__124.y&&__127.y);
                __123.z = (__124.z&&__127.z);
                __123.w = (__124.w&&__127.w);
              ushort4 __129;
                ushort4 __130;
                  int4 __131 = make_int4(2, 2, 2, 2);
                  int4 __132 = make_int4(0, 0, 0, 0);
                  __130.x = (__131.x<__132.x);
                  __130.y = (__131.y<__132.y);
                  __130.z = (__131.z<__132.z);
                  __130.w = (__131.w<__132.w);
                ushort4 __133;
                  int4 __134 = make_int4(0, 0, 0, 0);
                  __133.x = (__115.x<=__134.x);
                  __133.y = (__115.y<=__134.y);
                  __133.z = (__115.z<=__134.z);
                  __133.w = (__115.w<=__134.w);
                __129.x = (__130.x&&__133.x);
                __129.y = (__130.y&&__133.y);
                __129.z = (__130.z&&__133.z);
                __129.w = (__130.w&&__133.w);
              __122.x = (__123.x||__129.x);
              __122.y = (__123.y||__129.y);
              __122.z = (__123.z||__129.z);
              __122.w = (__123.w||__129.w);
            int4 __135;
              int4 __136 = make_int4(1, 1, 1, 1);
              __135.x = (__118.x-__136.x);
              __135.y = (__118.y-__136.y);
              __135.z = (__118.z-__136.z);
              __135.w = (__118.w-__136.w);
            __121.x = (bool(__122.x)?__118.x:__135.x);
            __121.y = (bool(__122.y)?__118.y:__135.y);
            __121.z = (bool(__122.z)?__118.z:__135.z);
            __121.w = (bool(__122.w)?__118.w:__135.w);
            __113.x = (__114.x+__121.x);
            __113.y = (__114.y+__121.y);
            __113.z = (__114.z+__121.z);
            __113.w = (__114.w+__121.w);
          int __137 = ((0x000000ff << 0) & (B[__113.x] << 0))|((0x000000ff << 8) & (B[__113.y] << 8))|((0x000000ff << 16) & (B[__113.z] << 16))|((0x000000ff << 24) & (B[__113.w] << 24));
          int __138;
          int4 __139;
            int4 __140;
              int4 __141 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __142 = make_int4(2, 2, 2, 2);
              __140.x = (__141.x%__142.x);
              __140.y = (__141.y%__142.y);
              __140.z = (__141.z%__142.z);
              __140.w = (__141.w%__142.w);
            int4 __143;
            ushort4 __144;
              ushort4 __145;
                ushort4 __146;
                  int4 __147 = make_int4(2, 2, 2, 2);
                  int4 __148 = make_int4(0, 0, 0, 0);
                  __146.x = (__147.x>=__148.x);
                  __146.y = (__147.y>=__148.y);
                  __146.z = (__147.z>=__148.z);
                  __146.w = (__147.w>=__148.w);
                ushort4 __149;
                  int4 __150 = make_int4(0, 0, 0, 0);
                  __149.x = (__140.x>=__150.x);
                  __149.y = (__140.y>=__150.y);
                  __149.z = (__140.z>=__150.z);
                  __149.w = (__140.w>=__150.w);
                __145.x = (__146.x&&__149.x);
                __145.y = (__146.y&&__149.y);
                __145.z = (__146.z&&__149.z);
                __145.w = (__146.w&&__149.w);
              ushort4 __151;
                ushort4 __152;
                  int4 __153 = make_int4(2, 2, 2, 2);
                  int4 __154 = make_int4(0, 0, 0, 0);
                  __152.x = (__153.x<__154.x);
                  __152.y = (__153.y<__154.y);
                  __152.z = (__153.z<__154.z);
                  __152.w = (__153.w<__154.w);
                ushort4 __155;
                  int4 __156 = make_int4(0, 0, 0, 0);
                  __155.x = (__140.x<=__156.x);
                  __155.y = (__140.y<=__156.y);
                  __155.z = (__140.z<=__156.z);
                  __155.w = (__140.w<=__156.w);
                __151.x = (__152.x&&__155.x);
                __151.y = (__152.y&&__155.y);
                __151.z = (__152.z&&__155.z);
                __151.w = (__152.w&&__155.w);
              __144.x = (__145.x||__151.x);
              __144.y = (__145.y||__151.y);
              __144.z = (__145.z||__151.z);
              __144.w = (__145.w||__151.w);
            int4 __157;
              int4 __158 = make_int4(2, 2, 2, 2);
              __157.x = (__140.x+__158.x);
              __157.y = (__140.y+__158.y);
              __157.z = (__140.z+__158.z);
              __157.w = (__140.w+__158.w);
            __143.x = (bool(__144.x)?__140.x:__157.x);
            __143.y = (bool(__144.y)?__140.y:__157.y);
            __143.z = (bool(__144.z)?__140.z:__157.z);
            __143.w = (bool(__144.w)?__140.w:__157.w);
            int4 __159 = make_int4(4, 4, 4, 4);
            __139.x = (__143.x*__159.x);
            __139.y = (__143.y*__159.y);
            __139.z = (__143.z*__159.z);
            __139.w = (__143.w*__159.w);
          __138=((signed char)(__139.x) << 0);
          __138=__138 & ~(0x000000ff << 8) |((signed char)(__139.y) << 8);
          __138=__138 & ~(0x000000ff << 16) |((signed char)(__139.z) << 16);
          __138=__138 & ~(0x000000ff << 24) |((signed char)(__139.w) << 24);
          __112=((((char)(__137 >> 0)) >> ((char)(__138 >> 0))) << 0);
          __112=__112 & ~(0x000000ff << 8) |((((char)(__137 >> 8)) >> ((char)(__138 >> 8))) << 8);
          __112=__112 & ~(0x000000ff << 16) |((((char)(__137 >> 16)) >> ((char)(__138 >> 16))) << 16);
          __112=__112 & ~(0x000000ff << 24) |((((char)(__137 >> 24)) >> ((char)(__138 >> 24))) << 24);
        int __160 = (int)252645135;
        __111=((((char)(__112 >> 0)) & ((char)(__160 >> 0))) << 0);
        __111=__111 & ~(0x000000ff << 8) |((((char)(__112 >> 8)) & ((char)(__160 >> 8))) << 8);
        __111=__111 & ~(0x000000ff << 16) |((((char)(__112 >> 16)) & ((char)(__160 >> 16))) << 16);
        __111=__111 & ~(0x000000ff << 24) |((((char)(__112 >> 24)) & ((char)(__160 >> 24))) << 24);
      __110.x = (int)(((char)(__111 >> 0)));
      __110.y = (int)(((char)(__111 >> 8)));
      __110.z = (int)(((char)(__111 >> 16)));
      __110.w = (int)(((char)(__111 >> 24)));
      uint2 __161 = make_uint2(__pack_half2(LUT[__110.x],LUT[__110.y]),__pack_half2(LUT[__110.z],LUT[__110.w]));
      uint2 __162 = make_uint2(__pack_half2(Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 8)], Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 8)]), __pack_half2(Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 8)], Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 8)]));
      ((half2*)(&(__109.x)))->x = (((half2*)(&(__161.x)))->x*((half2*)(&(__162.x)))->x);
      ((half2*)(&(__109.x)))->y = (((half2*)(&(__161.x)))->y*((half2*)(&(__162.x)))->y);
      ((half2*)(&(__109.y)))->x = (((half2*)(&(__161.y)))->x*((half2*)(&(__162.y)))->x);
      ((half2*)(&(__109.y)))->y = (((half2*)(&(__161.y)))->y*((half2*)(&(__162.y)))->y);
    *(uint2*)(B_decode_shared + ((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 4)) + 320)) = __109;
    uint2 __163;
      int4 __164;
      int __165;
        int __166;
          int4 __167;
            int4 __168 = make_int4(((((((((int)blockIdx.x) % 160) * 81920) + ((((int)threadIdx.x) >> 3) * 2560)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 30720), ((((((((int)blockIdx.x) % 160) * 81920) + ((((int)threadIdx.x) >> 3) * 2560)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 30720), ((((((((int)blockIdx.x) % 160) * 81920) + ((((int)threadIdx.x) >> 3) * 2560)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 30720), ((((((((int)blockIdx.x) % 160) * 81920) + ((((int)threadIdx.x) >> 3) * 2560)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 30720));
            int4 __169;
              int4 __170 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __171 = make_int4(2, 2, 2, 2);
              __169.x = (__170.x%__171.x);
              __169.y = (__170.y%__171.y);
              __169.z = (__170.z%__171.z);
              __169.w = (__170.w%__171.w);
            int4 __172;
              int4 __173 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __174 = make_int4(2, 2, 2, 2);
              __172.x = (__173.x/__174.x);
              __172.y = (__173.y/__174.y);
              __172.z = (__173.z/__174.z);
              __172.w = (__173.w/__174.w);
            int4 __175;
            ushort4 __176;
              ushort4 __177;
                ushort4 __178;
                  int4 __179 = make_int4(2, 2, 2, 2);
                  int4 __180 = make_int4(0, 0, 0, 0);
                  __178.x = (__179.x>=__180.x);
                  __178.y = (__179.y>=__180.y);
                  __178.z = (__179.z>=__180.z);
                  __178.w = (__179.w>=__180.w);
                ushort4 __181;
                  int4 __182 = make_int4(0, 0, 0, 0);
                  __181.x = (__169.x>=__182.x);
                  __181.y = (__169.y>=__182.y);
                  __181.z = (__169.z>=__182.z);
                  __181.w = (__169.w>=__182.w);
                __177.x = (__178.x&&__181.x);
                __177.y = (__178.y&&__181.y);
                __177.z = (__178.z&&__181.z);
                __177.w = (__178.w&&__181.w);
              ushort4 __183;
                ushort4 __184;
                  int4 __185 = make_int4(2, 2, 2, 2);
                  int4 __186 = make_int4(0, 0, 0, 0);
                  __184.x = (__185.x<__186.x);
                  __184.y = (__185.y<__186.y);
                  __184.z = (__185.z<__186.z);
                  __184.w = (__185.w<__186.w);
                ushort4 __187;
                  int4 __188 = make_int4(0, 0, 0, 0);
                  __187.x = (__169.x<=__188.x);
                  __187.y = (__169.y<=__188.y);
                  __187.z = (__169.z<=__188.z);
                  __187.w = (__169.w<=__188.w);
                __183.x = (__184.x&&__187.x);
                __183.y = (__184.y&&__187.y);
                __183.z = (__184.z&&__187.z);
                __183.w = (__184.w&&__187.w);
              __176.x = (__177.x||__183.x);
              __176.y = (__177.y||__183.y);
              __176.z = (__177.z||__183.z);
              __176.w = (__177.w||__183.w);
            int4 __189;
              int4 __190 = make_int4(1, 1, 1, 1);
              __189.x = (__172.x-__190.x);
              __189.y = (__172.y-__190.y);
              __189.z = (__172.z-__190.z);
              __189.w = (__172.w-__190.w);
            __175.x = (bool(__176.x)?__172.x:__189.x);
            __175.y = (bool(__176.y)?__172.y:__189.y);
            __175.z = (bool(__176.z)?__172.z:__189.z);
            __175.w = (bool(__176.w)?__172.w:__189.w);
            __167.x = (__168.x+__175.x);
            __167.y = (__168.y+__175.y);
            __167.z = (__168.z+__175.z);
            __167.w = (__168.w+__175.w);
          int __191 = ((0x000000ff << 0) & (B[__167.x] << 0))|((0x000000ff << 8) & (B[__167.y] << 8))|((0x000000ff << 16) & (B[__167.z] << 16))|((0x000000ff << 24) & (B[__167.w] << 24));
          int __192;
          int4 __193;
            int4 __194;
              int4 __195 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __196 = make_int4(2, 2, 2, 2);
              __194.x = (__195.x%__196.x);
              __194.y = (__195.y%__196.y);
              __194.z = (__195.z%__196.z);
              __194.w = (__195.w%__196.w);
            int4 __197;
            ushort4 __198;
              ushort4 __199;
                ushort4 __200;
                  int4 __201 = make_int4(2, 2, 2, 2);
                  int4 __202 = make_int4(0, 0, 0, 0);
                  __200.x = (__201.x>=__202.x);
                  __200.y = (__201.y>=__202.y);
                  __200.z = (__201.z>=__202.z);
                  __200.w = (__201.w>=__202.w);
                ushort4 __203;
                  int4 __204 = make_int4(0, 0, 0, 0);
                  __203.x = (__194.x>=__204.x);
                  __203.y = (__194.y>=__204.y);
                  __203.z = (__194.z>=__204.z);
                  __203.w = (__194.w>=__204.w);
                __199.x = (__200.x&&__203.x);
                __199.y = (__200.y&&__203.y);
                __199.z = (__200.z&&__203.z);
                __199.w = (__200.w&&__203.w);
              ushort4 __205;
                ushort4 __206;
                  int4 __207 = make_int4(2, 2, 2, 2);
                  int4 __208 = make_int4(0, 0, 0, 0);
                  __206.x = (__207.x<__208.x);
                  __206.y = (__207.y<__208.y);
                  __206.z = (__207.z<__208.z);
                  __206.w = (__207.w<__208.w);
                ushort4 __209;
                  int4 __210 = make_int4(0, 0, 0, 0);
                  __209.x = (__194.x<=__210.x);
                  __209.y = (__194.y<=__210.y);
                  __209.z = (__194.z<=__210.z);
                  __209.w = (__194.w<=__210.w);
                __205.x = (__206.x&&__209.x);
                __205.y = (__206.y&&__209.y);
                __205.z = (__206.z&&__209.z);
                __205.w = (__206.w&&__209.w);
              __198.x = (__199.x||__205.x);
              __198.y = (__199.y||__205.y);
              __198.z = (__199.z||__205.z);
              __198.w = (__199.w||__205.w);
            int4 __211;
              int4 __212 = make_int4(2, 2, 2, 2);
              __211.x = (__194.x+__212.x);
              __211.y = (__194.y+__212.y);
              __211.z = (__194.z+__212.z);
              __211.w = (__194.w+__212.w);
            __197.x = (bool(__198.x)?__194.x:__211.x);
            __197.y = (bool(__198.y)?__194.y:__211.y);
            __197.z = (bool(__198.z)?__194.z:__211.z);
            __197.w = (bool(__198.w)?__194.w:__211.w);
            int4 __213 = make_int4(4, 4, 4, 4);
            __193.x = (__197.x*__213.x);
            __193.y = (__197.y*__213.y);
            __193.z = (__197.z*__213.z);
            __193.w = (__197.w*__213.w);
          __192=((signed char)(__193.x) << 0);
          __192=__192 & ~(0x000000ff << 8) |((signed char)(__193.y) << 8);
          __192=__192 & ~(0x000000ff << 16) |((signed char)(__193.z) << 16);
          __192=__192 & ~(0x000000ff << 24) |((signed char)(__193.w) << 24);
          __166=((((char)(__191 >> 0)) >> ((char)(__192 >> 0))) << 0);
          __166=__166 & ~(0x000000ff << 8) |((((char)(__191 >> 8)) >> ((char)(__192 >> 8))) << 8);
          __166=__166 & ~(0x000000ff << 16) |((((char)(__191 >> 16)) >> ((char)(__192 >> 16))) << 16);
          __166=__166 & ~(0x000000ff << 24) |((((char)(__191 >> 24)) >> ((char)(__192 >> 24))) << 24);
        int __214 = (int)252645135;
        __165=((((char)(__166 >> 0)) & ((char)(__214 >> 0))) << 0);
        __165=__165 & ~(0x000000ff << 8) |((((char)(__166 >> 8)) & ((char)(__214 >> 8))) << 8);
        __165=__165 & ~(0x000000ff << 16) |((((char)(__166 >> 16)) & ((char)(__214 >> 16))) << 16);
        __165=__165 & ~(0x000000ff << 24) |((((char)(__166 >> 24)) & ((char)(__214 >> 24))) << 24);
      __164.x = (int)(((char)(__165 >> 0)));
      __164.y = (int)(((char)(__165 >> 8)));
      __164.z = (int)(((char)(__165 >> 16)));
      __164.w = (int)(((char)(__165 >> 24)));
      uint2 __215 = make_uint2(__pack_half2(LUT[__164.x],LUT[__164.y]),__pack_half2(LUT[__164.z],LUT[__164.w]));
      uint2 __216 = make_uint2(__pack_half2(Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 12)], Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 12)]), __pack_half2(Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 12)], Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 12)]));
      ((half2*)(&(__163.x)))->x = (((half2*)(&(__215.x)))->x*((half2*)(&(__216.x)))->x);
      ((half2*)(&(__163.x)))->y = (((half2*)(&(__215.x)))->y*((half2*)(&(__216.x)))->y);
      ((half2*)(&(__163.y)))->x = (((half2*)(&(__215.y)))->x*((half2*)(&(__216.y)))->x);
      ((half2*)(&(__163.y)))->y = (((half2*)(&(__215.y)))->y*((half2*)(&(__216.y)))->y);
    *(uint2*)(B_decode_shared + ((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 4)) + 480)) = __163;
    uint2 __217;
      int4 __218;
      int __219;
        int __220;
          int4 __221;
            int4 __222 = make_int4(((((((((int)blockIdx.x) % 160) * 81920) + ((((int)threadIdx.x) >> 3) * 2560)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 40960), ((((((((int)blockIdx.x) % 160) * 81920) + ((((int)threadIdx.x) >> 3) * 2560)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 40960), ((((((((int)blockIdx.x) % 160) * 81920) + ((((int)threadIdx.x) >> 3) * 2560)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 40960), ((((((((int)blockIdx.x) % 160) * 81920) + ((((int)threadIdx.x) >> 3) * 2560)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 40960));
            int4 __223;
              int4 __224 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __225 = make_int4(2, 2, 2, 2);
              __223.x = (__224.x%__225.x);
              __223.y = (__224.y%__225.y);
              __223.z = (__224.z%__225.z);
              __223.w = (__224.w%__225.w);
            int4 __226;
              int4 __227 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __228 = make_int4(2, 2, 2, 2);
              __226.x = (__227.x/__228.x);
              __226.y = (__227.y/__228.y);
              __226.z = (__227.z/__228.z);
              __226.w = (__227.w/__228.w);
            int4 __229;
            ushort4 __230;
              ushort4 __231;
                ushort4 __232;
                  int4 __233 = make_int4(2, 2, 2, 2);
                  int4 __234 = make_int4(0, 0, 0, 0);
                  __232.x = (__233.x>=__234.x);
                  __232.y = (__233.y>=__234.y);
                  __232.z = (__233.z>=__234.z);
                  __232.w = (__233.w>=__234.w);
                ushort4 __235;
                  int4 __236 = make_int4(0, 0, 0, 0);
                  __235.x = (__223.x>=__236.x);
                  __235.y = (__223.y>=__236.y);
                  __235.z = (__223.z>=__236.z);
                  __235.w = (__223.w>=__236.w);
                __231.x = (__232.x&&__235.x);
                __231.y = (__232.y&&__235.y);
                __231.z = (__232.z&&__235.z);
                __231.w = (__232.w&&__235.w);
              ushort4 __237;
                ushort4 __238;
                  int4 __239 = make_int4(2, 2, 2, 2);
                  int4 __240 = make_int4(0, 0, 0, 0);
                  __238.x = (__239.x<__240.x);
                  __238.y = (__239.y<__240.y);
                  __238.z = (__239.z<__240.z);
                  __238.w = (__239.w<__240.w);
                ushort4 __241;
                  int4 __242 = make_int4(0, 0, 0, 0);
                  __241.x = (__223.x<=__242.x);
                  __241.y = (__223.y<=__242.y);
                  __241.z = (__223.z<=__242.z);
                  __241.w = (__223.w<=__242.w);
                __237.x = (__238.x&&__241.x);
                __237.y = (__238.y&&__241.y);
                __237.z = (__238.z&&__241.z);
                __237.w = (__238.w&&__241.w);
              __230.x = (__231.x||__237.x);
              __230.y = (__231.y||__237.y);
              __230.z = (__231.z||__237.z);
              __230.w = (__231.w||__237.w);
            int4 __243;
              int4 __244 = make_int4(1, 1, 1, 1);
              __243.x = (__226.x-__244.x);
              __243.y = (__226.y-__244.y);
              __243.z = (__226.z-__244.z);
              __243.w = (__226.w-__244.w);
            __229.x = (bool(__230.x)?__226.x:__243.x);
            __229.y = (bool(__230.y)?__226.y:__243.y);
            __229.z = (bool(__230.z)?__226.z:__243.z);
            __229.w = (bool(__230.w)?__226.w:__243.w);
            __221.x = (__222.x+__229.x);
            __221.y = (__222.y+__229.y);
            __221.z = (__222.z+__229.z);
            __221.w = (__222.w+__229.w);
          int __245 = ((0x000000ff << 0) & (B[__221.x] << 0))|((0x000000ff << 8) & (B[__221.y] << 8))|((0x000000ff << 16) & (B[__221.z] << 16))|((0x000000ff << 24) & (B[__221.w] << 24));
          int __246;
          int4 __247;
            int4 __248;
              int4 __249 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __250 = make_int4(2, 2, 2, 2);
              __248.x = (__249.x%__250.x);
              __248.y = (__249.y%__250.y);
              __248.z = (__249.z%__250.z);
              __248.w = (__249.w%__250.w);
            int4 __251;
            ushort4 __252;
              ushort4 __253;
                ushort4 __254;
                  int4 __255 = make_int4(2, 2, 2, 2);
                  int4 __256 = make_int4(0, 0, 0, 0);
                  __254.x = (__255.x>=__256.x);
                  __254.y = (__255.y>=__256.y);
                  __254.z = (__255.z>=__256.z);
                  __254.w = (__255.w>=__256.w);
                ushort4 __257;
                  int4 __258 = make_int4(0, 0, 0, 0);
                  __257.x = (__248.x>=__258.x);
                  __257.y = (__248.y>=__258.y);
                  __257.z = (__248.z>=__258.z);
                  __257.w = (__248.w>=__258.w);
                __253.x = (__254.x&&__257.x);
                __253.y = (__254.y&&__257.y);
                __253.z = (__254.z&&__257.z);
                __253.w = (__254.w&&__257.w);
              ushort4 __259;
                ushort4 __260;
                  int4 __261 = make_int4(2, 2, 2, 2);
                  int4 __262 = make_int4(0, 0, 0, 0);
                  __260.x = (__261.x<__262.x);
                  __260.y = (__261.y<__262.y);
                  __260.z = (__261.z<__262.z);
                  __260.w = (__261.w<__262.w);
                ushort4 __263;
                  int4 __264 = make_int4(0, 0, 0, 0);
                  __263.x = (__248.x<=__264.x);
                  __263.y = (__248.y<=__264.y);
                  __263.z = (__248.z<=__264.z);
                  __263.w = (__248.w<=__264.w);
                __259.x = (__260.x&&__263.x);
                __259.y = (__260.y&&__263.y);
                __259.z = (__260.z&&__263.z);
                __259.w = (__260.w&&__263.w);
              __252.x = (__253.x||__259.x);
              __252.y = (__253.y||__259.y);
              __252.z = (__253.z||__259.z);
              __252.w = (__253.w||__259.w);
            int4 __265;
              int4 __266 = make_int4(2, 2, 2, 2);
              __265.x = (__248.x+__266.x);
              __265.y = (__248.y+__266.y);
              __265.z = (__248.z+__266.z);
              __265.w = (__248.w+__266.w);
            __251.x = (bool(__252.x)?__248.x:__265.x);
            __251.y = (bool(__252.y)?__248.y:__265.y);
            __251.z = (bool(__252.z)?__248.z:__265.z);
            __251.w = (bool(__252.w)?__248.w:__265.w);
            int4 __267 = make_int4(4, 4, 4, 4);
            __247.x = (__251.x*__267.x);
            __247.y = (__251.y*__267.y);
            __247.z = (__251.z*__267.z);
            __247.w = (__251.w*__267.w);
          __246=((signed char)(__247.x) << 0);
          __246=__246 & ~(0x000000ff << 8) |((signed char)(__247.y) << 8);
          __246=__246 & ~(0x000000ff << 16) |((signed char)(__247.z) << 16);
          __246=__246 & ~(0x000000ff << 24) |((signed char)(__247.w) << 24);
          __220=((((char)(__245 >> 0)) >> ((char)(__246 >> 0))) << 0);
          __220=__220 & ~(0x000000ff << 8) |((((char)(__245 >> 8)) >> ((char)(__246 >> 8))) << 8);
          __220=__220 & ~(0x000000ff << 16) |((((char)(__245 >> 16)) >> ((char)(__246 >> 16))) << 16);
          __220=__220 & ~(0x000000ff << 24) |((((char)(__245 >> 24)) >> ((char)(__246 >> 24))) << 24);
        int __268 = (int)252645135;
        __219=((((char)(__220 >> 0)) & ((char)(__268 >> 0))) << 0);
        __219=__219 & ~(0x000000ff << 8) |((((char)(__220 >> 8)) & ((char)(__268 >> 8))) << 8);
        __219=__219 & ~(0x000000ff << 16) |((((char)(__220 >> 16)) & ((char)(__268 >> 16))) << 16);
        __219=__219 & ~(0x000000ff << 24) |((((char)(__220 >> 24)) & ((char)(__268 >> 24))) << 24);
      __218.x = (int)(((char)(__219 >> 0)));
      __218.y = (int)(((char)(__219 >> 8)));
      __218.z = (int)(((char)(__219 >> 16)));
      __218.w = (int)(((char)(__219 >> 24)));
      uint2 __269 = make_uint2(__pack_half2(LUT[__218.x],LUT[__218.y]),__pack_half2(LUT[__218.z],LUT[__218.w]));
      uint2 __270 = make_uint2(__pack_half2(Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 16)], Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 16)]), __pack_half2(Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 16)], Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 16)]));
      ((half2*)(&(__217.x)))->x = (((half2*)(&(__269.x)))->x*((half2*)(&(__270.x)))->x);
      ((half2*)(&(__217.x)))->y = (((half2*)(&(__269.x)))->y*((half2*)(&(__270.x)))->y);
      ((half2*)(&(__217.y)))->x = (((half2*)(&(__269.y)))->x*((half2*)(&(__270.y)))->x);
      ((half2*)(&(__217.y)))->y = (((half2*)(&(__269.y)))->y*((half2*)(&(__270.y)))->y);
    *(uint2*)(B_decode_shared + ((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 4)) + 640)) = __217;
    uint2 __271;
      int4 __272;
      int __273;
        int __274;
          int4 __275;
            int4 __276 = make_int4(((((((((int)blockIdx.x) % 160) * 81920) + ((((int)threadIdx.x) >> 3) * 2560)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 51200), ((((((((int)blockIdx.x) % 160) * 81920) + ((((int)threadIdx.x) >> 3) * 2560)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 51200), ((((((((int)blockIdx.x) % 160) * 81920) + ((((int)threadIdx.x) >> 3) * 2560)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 51200), ((((((((int)blockIdx.x) % 160) * 81920) + ((((int)threadIdx.x) >> 3) * 2560)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 51200));
            int4 __277;
              int4 __278 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __279 = make_int4(2, 2, 2, 2);
              __277.x = (__278.x%__279.x);
              __277.y = (__278.y%__279.y);
              __277.z = (__278.z%__279.z);
              __277.w = (__278.w%__279.w);
            int4 __280;
              int4 __281 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __282 = make_int4(2, 2, 2, 2);
              __280.x = (__281.x/__282.x);
              __280.y = (__281.y/__282.y);
              __280.z = (__281.z/__282.z);
              __280.w = (__281.w/__282.w);
            int4 __283;
            ushort4 __284;
              ushort4 __285;
                ushort4 __286;
                  int4 __287 = make_int4(2, 2, 2, 2);
                  int4 __288 = make_int4(0, 0, 0, 0);
                  __286.x = (__287.x>=__288.x);
                  __286.y = (__287.y>=__288.y);
                  __286.z = (__287.z>=__288.z);
                  __286.w = (__287.w>=__288.w);
                ushort4 __289;
                  int4 __290 = make_int4(0, 0, 0, 0);
                  __289.x = (__277.x>=__290.x);
                  __289.y = (__277.y>=__290.y);
                  __289.z = (__277.z>=__290.z);
                  __289.w = (__277.w>=__290.w);
                __285.x = (__286.x&&__289.x);
                __285.y = (__286.y&&__289.y);
                __285.z = (__286.z&&__289.z);
                __285.w = (__286.w&&__289.w);
              ushort4 __291;
                ushort4 __292;
                  int4 __293 = make_int4(2, 2, 2, 2);
                  int4 __294 = make_int4(0, 0, 0, 0);
                  __292.x = (__293.x<__294.x);
                  __292.y = (__293.y<__294.y);
                  __292.z = (__293.z<__294.z);
                  __292.w = (__293.w<__294.w);
                ushort4 __295;
                  int4 __296 = make_int4(0, 0, 0, 0);
                  __295.x = (__277.x<=__296.x);
                  __295.y = (__277.y<=__296.y);
                  __295.z = (__277.z<=__296.z);
                  __295.w = (__277.w<=__296.w);
                __291.x = (__292.x&&__295.x);
                __291.y = (__292.y&&__295.y);
                __291.z = (__292.z&&__295.z);
                __291.w = (__292.w&&__295.w);
              __284.x = (__285.x||__291.x);
              __284.y = (__285.y||__291.y);
              __284.z = (__285.z||__291.z);
              __284.w = (__285.w||__291.w);
            int4 __297;
              int4 __298 = make_int4(1, 1, 1, 1);
              __297.x = (__280.x-__298.x);
              __297.y = (__280.y-__298.y);
              __297.z = (__280.z-__298.z);
              __297.w = (__280.w-__298.w);
            __283.x = (bool(__284.x)?__280.x:__297.x);
            __283.y = (bool(__284.y)?__280.y:__297.y);
            __283.z = (bool(__284.z)?__280.z:__297.z);
            __283.w = (bool(__284.w)?__280.w:__297.w);
            __275.x = (__276.x+__283.x);
            __275.y = (__276.y+__283.y);
            __275.z = (__276.z+__283.z);
            __275.w = (__276.w+__283.w);
          int __299 = ((0x000000ff << 0) & (B[__275.x] << 0))|((0x000000ff << 8) & (B[__275.y] << 8))|((0x000000ff << 16) & (B[__275.z] << 16))|((0x000000ff << 24) & (B[__275.w] << 24));
          int __300;
          int4 __301;
            int4 __302;
              int4 __303 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __304 = make_int4(2, 2, 2, 2);
              __302.x = (__303.x%__304.x);
              __302.y = (__303.y%__304.y);
              __302.z = (__303.z%__304.z);
              __302.w = (__303.w%__304.w);
            int4 __305;
            ushort4 __306;
              ushort4 __307;
                ushort4 __308;
                  int4 __309 = make_int4(2, 2, 2, 2);
                  int4 __310 = make_int4(0, 0, 0, 0);
                  __308.x = (__309.x>=__310.x);
                  __308.y = (__309.y>=__310.y);
                  __308.z = (__309.z>=__310.z);
                  __308.w = (__309.w>=__310.w);
                ushort4 __311;
                  int4 __312 = make_int4(0, 0, 0, 0);
                  __311.x = (__302.x>=__312.x);
                  __311.y = (__302.y>=__312.y);
                  __311.z = (__302.z>=__312.z);
                  __311.w = (__302.w>=__312.w);
                __307.x = (__308.x&&__311.x);
                __307.y = (__308.y&&__311.y);
                __307.z = (__308.z&&__311.z);
                __307.w = (__308.w&&__311.w);
              ushort4 __313;
                ushort4 __314;
                  int4 __315 = make_int4(2, 2, 2, 2);
                  int4 __316 = make_int4(0, 0, 0, 0);
                  __314.x = (__315.x<__316.x);
                  __314.y = (__315.y<__316.y);
                  __314.z = (__315.z<__316.z);
                  __314.w = (__315.w<__316.w);
                ushort4 __317;
                  int4 __318 = make_int4(0, 0, 0, 0);
                  __317.x = (__302.x<=__318.x);
                  __317.y = (__302.y<=__318.y);
                  __317.z = (__302.z<=__318.z);
                  __317.w = (__302.w<=__318.w);
                __313.x = (__314.x&&__317.x);
                __313.y = (__314.y&&__317.y);
                __313.z = (__314.z&&__317.z);
                __313.w = (__314.w&&__317.w);
              __306.x = (__307.x||__313.x);
              __306.y = (__307.y||__313.y);
              __306.z = (__307.z||__313.z);
              __306.w = (__307.w||__313.w);
            int4 __319;
              int4 __320 = make_int4(2, 2, 2, 2);
              __319.x = (__302.x+__320.x);
              __319.y = (__302.y+__320.y);
              __319.z = (__302.z+__320.z);
              __319.w = (__302.w+__320.w);
            __305.x = (bool(__306.x)?__302.x:__319.x);
            __305.y = (bool(__306.y)?__302.y:__319.y);
            __305.z = (bool(__306.z)?__302.z:__319.z);
            __305.w = (bool(__306.w)?__302.w:__319.w);
            int4 __321 = make_int4(4, 4, 4, 4);
            __301.x = (__305.x*__321.x);
            __301.y = (__305.y*__321.y);
            __301.z = (__305.z*__321.z);
            __301.w = (__305.w*__321.w);
          __300=((signed char)(__301.x) << 0);
          __300=__300 & ~(0x000000ff << 8) |((signed char)(__301.y) << 8);
          __300=__300 & ~(0x000000ff << 16) |((signed char)(__301.z) << 16);
          __300=__300 & ~(0x000000ff << 24) |((signed char)(__301.w) << 24);
          __274=((((char)(__299 >> 0)) >> ((char)(__300 >> 0))) << 0);
          __274=__274 & ~(0x000000ff << 8) |((((char)(__299 >> 8)) >> ((char)(__300 >> 8))) << 8);
          __274=__274 & ~(0x000000ff << 16) |((((char)(__299 >> 16)) >> ((char)(__300 >> 16))) << 16);
          __274=__274 & ~(0x000000ff << 24) |((((char)(__299 >> 24)) >> ((char)(__300 >> 24))) << 24);
        int __322 = (int)252645135;
        __273=((((char)(__274 >> 0)) & ((char)(__322 >> 0))) << 0);
        __273=__273 & ~(0x000000ff << 8) |((((char)(__274 >> 8)) & ((char)(__322 >> 8))) << 8);
        __273=__273 & ~(0x000000ff << 16) |((((char)(__274 >> 16)) & ((char)(__322 >> 16))) << 16);
        __273=__273 & ~(0x000000ff << 24) |((((char)(__274 >> 24)) & ((char)(__322 >> 24))) << 24);
      __272.x = (int)(((char)(__273 >> 0)));
      __272.y = (int)(((char)(__273 >> 8)));
      __272.z = (int)(((char)(__273 >> 16)));
      __272.w = (int)(((char)(__273 >> 24)));
      uint2 __323 = make_uint2(__pack_half2(LUT[__272.x],LUT[__272.y]),__pack_half2(LUT[__272.z],LUT[__272.w]));
      uint2 __324 = make_uint2(__pack_half2(Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 20)], Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 20)]), __pack_half2(Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 20)], Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 20)]));
      ((half2*)(&(__271.x)))->x = (((half2*)(&(__323.x)))->x*((half2*)(&(__324.x)))->x);
      ((half2*)(&(__271.x)))->y = (((half2*)(&(__323.x)))->y*((half2*)(&(__324.x)))->y);
      ((half2*)(&(__271.y)))->x = (((half2*)(&(__323.y)))->x*((half2*)(&(__324.y)))->x);
      ((half2*)(&(__271.y)))->y = (((half2*)(&(__323.y)))->y*((half2*)(&(__324.y)))->y);
    *(uint2*)(B_decode_shared + ((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 4)) + 800)) = __271;
    uint2 __325;
      int4 __326;
      int __327;
        int __328;
          int4 __329;
            int4 __330 = make_int4(((((((((int)blockIdx.x) % 160) * 81920) + ((((int)threadIdx.x) >> 3) * 2560)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 61440), ((((((((int)blockIdx.x) % 160) * 81920) + ((((int)threadIdx.x) >> 3) * 2560)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 61440), ((((((((int)blockIdx.x) % 160) * 81920) + ((((int)threadIdx.x) >> 3) * 2560)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 61440), ((((((((int)blockIdx.x) % 160) * 81920) + ((((int)threadIdx.x) >> 3) * 2560)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 61440));
            int4 __331;
              int4 __332 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __333 = make_int4(2, 2, 2, 2);
              __331.x = (__332.x%__333.x);
              __331.y = (__332.y%__333.y);
              __331.z = (__332.z%__333.z);
              __331.w = (__332.w%__333.w);
            int4 __334;
              int4 __335 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __336 = make_int4(2, 2, 2, 2);
              __334.x = (__335.x/__336.x);
              __334.y = (__335.y/__336.y);
              __334.z = (__335.z/__336.z);
              __334.w = (__335.w/__336.w);
            int4 __337;
            ushort4 __338;
              ushort4 __339;
                ushort4 __340;
                  int4 __341 = make_int4(2, 2, 2, 2);
                  int4 __342 = make_int4(0, 0, 0, 0);
                  __340.x = (__341.x>=__342.x);
                  __340.y = (__341.y>=__342.y);
                  __340.z = (__341.z>=__342.z);
                  __340.w = (__341.w>=__342.w);
                ushort4 __343;
                  int4 __344 = make_int4(0, 0, 0, 0);
                  __343.x = (__331.x>=__344.x);
                  __343.y = (__331.y>=__344.y);
                  __343.z = (__331.z>=__344.z);
                  __343.w = (__331.w>=__344.w);
                __339.x = (__340.x&&__343.x);
                __339.y = (__340.y&&__343.y);
                __339.z = (__340.z&&__343.z);
                __339.w = (__340.w&&__343.w);
              ushort4 __345;
                ushort4 __346;
                  int4 __347 = make_int4(2, 2, 2, 2);
                  int4 __348 = make_int4(0, 0, 0, 0);
                  __346.x = (__347.x<__348.x);
                  __346.y = (__347.y<__348.y);
                  __346.z = (__347.z<__348.z);
                  __346.w = (__347.w<__348.w);
                ushort4 __349;
                  int4 __350 = make_int4(0, 0, 0, 0);
                  __349.x = (__331.x<=__350.x);
                  __349.y = (__331.y<=__350.y);
                  __349.z = (__331.z<=__350.z);
                  __349.w = (__331.w<=__350.w);
                __345.x = (__346.x&&__349.x);
                __345.y = (__346.y&&__349.y);
                __345.z = (__346.z&&__349.z);
                __345.w = (__346.w&&__349.w);
              __338.x = (__339.x||__345.x);
              __338.y = (__339.y||__345.y);
              __338.z = (__339.z||__345.z);
              __338.w = (__339.w||__345.w);
            int4 __351;
              int4 __352 = make_int4(1, 1, 1, 1);
              __351.x = (__334.x-__352.x);
              __351.y = (__334.y-__352.y);
              __351.z = (__334.z-__352.z);
              __351.w = (__334.w-__352.w);
            __337.x = (bool(__338.x)?__334.x:__351.x);
            __337.y = (bool(__338.y)?__334.y:__351.y);
            __337.z = (bool(__338.z)?__334.z:__351.z);
            __337.w = (bool(__338.w)?__334.w:__351.w);
            __329.x = (__330.x+__337.x);
            __329.y = (__330.y+__337.y);
            __329.z = (__330.z+__337.z);
            __329.w = (__330.w+__337.w);
          int __353 = ((0x000000ff << 0) & (B[__329.x] << 0))|((0x000000ff << 8) & (B[__329.y] << 8))|((0x000000ff << 16) & (B[__329.z] << 16))|((0x000000ff << 24) & (B[__329.w] << 24));
          int __354;
          int4 __355;
            int4 __356;
              int4 __357 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __358 = make_int4(2, 2, 2, 2);
              __356.x = (__357.x%__358.x);
              __356.y = (__357.y%__358.y);
              __356.z = (__357.z%__358.z);
              __356.w = (__357.w%__358.w);
            int4 __359;
            ushort4 __360;
              ushort4 __361;
                ushort4 __362;
                  int4 __363 = make_int4(2, 2, 2, 2);
                  int4 __364 = make_int4(0, 0, 0, 0);
                  __362.x = (__363.x>=__364.x);
                  __362.y = (__363.y>=__364.y);
                  __362.z = (__363.z>=__364.z);
                  __362.w = (__363.w>=__364.w);
                ushort4 __365;
                  int4 __366 = make_int4(0, 0, 0, 0);
                  __365.x = (__356.x>=__366.x);
                  __365.y = (__356.y>=__366.y);
                  __365.z = (__356.z>=__366.z);
                  __365.w = (__356.w>=__366.w);
                __361.x = (__362.x&&__365.x);
                __361.y = (__362.y&&__365.y);
                __361.z = (__362.z&&__365.z);
                __361.w = (__362.w&&__365.w);
              ushort4 __367;
                ushort4 __368;
                  int4 __369 = make_int4(2, 2, 2, 2);
                  int4 __370 = make_int4(0, 0, 0, 0);
                  __368.x = (__369.x<__370.x);
                  __368.y = (__369.y<__370.y);
                  __368.z = (__369.z<__370.z);
                  __368.w = (__369.w<__370.w);
                ushort4 __371;
                  int4 __372 = make_int4(0, 0, 0, 0);
                  __371.x = (__356.x<=__372.x);
                  __371.y = (__356.y<=__372.y);
                  __371.z = (__356.z<=__372.z);
                  __371.w = (__356.w<=__372.w);
                __367.x = (__368.x&&__371.x);
                __367.y = (__368.y&&__371.y);
                __367.z = (__368.z&&__371.z);
                __367.w = (__368.w&&__371.w);
              __360.x = (__361.x||__367.x);
              __360.y = (__361.y||__367.y);
              __360.z = (__361.z||__367.z);
              __360.w = (__361.w||__367.w);
            int4 __373;
              int4 __374 = make_int4(2, 2, 2, 2);
              __373.x = (__356.x+__374.x);
              __373.y = (__356.y+__374.y);
              __373.z = (__356.z+__374.z);
              __373.w = (__356.w+__374.w);
            __359.x = (bool(__360.x)?__356.x:__373.x);
            __359.y = (bool(__360.y)?__356.y:__373.y);
            __359.z = (bool(__360.z)?__356.z:__373.z);
            __359.w = (bool(__360.w)?__356.w:__373.w);
            int4 __375 = make_int4(4, 4, 4, 4);
            __355.x = (__359.x*__375.x);
            __355.y = (__359.y*__375.y);
            __355.z = (__359.z*__375.z);
            __355.w = (__359.w*__375.w);
          __354=((signed char)(__355.x) << 0);
          __354=__354 & ~(0x000000ff << 8) |((signed char)(__355.y) << 8);
          __354=__354 & ~(0x000000ff << 16) |((signed char)(__355.z) << 16);
          __354=__354 & ~(0x000000ff << 24) |((signed char)(__355.w) << 24);
          __328=((((char)(__353 >> 0)) >> ((char)(__354 >> 0))) << 0);
          __328=__328 & ~(0x000000ff << 8) |((((char)(__353 >> 8)) >> ((char)(__354 >> 8))) << 8);
          __328=__328 & ~(0x000000ff << 16) |((((char)(__353 >> 16)) >> ((char)(__354 >> 16))) << 16);
          __328=__328 & ~(0x000000ff << 24) |((((char)(__353 >> 24)) >> ((char)(__354 >> 24))) << 24);
        int __376 = (int)252645135;
        __327=((((char)(__328 >> 0)) & ((char)(__376 >> 0))) << 0);
        __327=__327 & ~(0x000000ff << 8) |((((char)(__328 >> 8)) & ((char)(__376 >> 8))) << 8);
        __327=__327 & ~(0x000000ff << 16) |((((char)(__328 >> 16)) & ((char)(__376 >> 16))) << 16);
        __327=__327 & ~(0x000000ff << 24) |((((char)(__328 >> 24)) & ((char)(__376 >> 24))) << 24);
      __326.x = (int)(((char)(__327 >> 0)));
      __326.y = (int)(((char)(__327 >> 8)));
      __326.z = (int)(((char)(__327 >> 16)));
      __326.w = (int)(((char)(__327 >> 24)));
      uint2 __377 = make_uint2(__pack_half2(LUT[__326.x],LUT[__326.y]),__pack_half2(LUT[__326.z],LUT[__326.w]));
      uint2 __378 = make_uint2(__pack_half2(Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 24)], Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 24)]), __pack_half2(Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 24)], Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 24)]));
      ((half2*)(&(__325.x)))->x = (((half2*)(&(__377.x)))->x*((half2*)(&(__378.x)))->x);
      ((half2*)(&(__325.x)))->y = (((half2*)(&(__377.x)))->y*((half2*)(&(__378.x)))->y);
      ((half2*)(&(__325.y)))->x = (((half2*)(&(__377.y)))->x*((half2*)(&(__378.y)))->x);
      ((half2*)(&(__325.y)))->y = (((half2*)(&(__377.y)))->y*((half2*)(&(__378.y)))->y);
    *(uint2*)(B_decode_shared + ((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 4)) + 960)) = __325;
    uint2 __379;
      int4 __380;
      int __381;
        int __382;
          int4 __383;
            int4 __384 = make_int4(((((((((int)blockIdx.x) % 160) * 81920) + ((((int)threadIdx.x) >> 3) * 2560)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 71680), ((((((((int)blockIdx.x) % 160) * 81920) + ((((int)threadIdx.x) >> 3) * 2560)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 71680), ((((((((int)blockIdx.x) % 160) * 81920) + ((((int)threadIdx.x) >> 3) * 2560)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 71680), ((((((((int)blockIdx.x) % 160) * 81920) + ((((int)threadIdx.x) >> 3) * 2560)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 71680));
            int4 __385;
              int4 __386 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __387 = make_int4(2, 2, 2, 2);
              __385.x = (__386.x%__387.x);
              __385.y = (__386.y%__387.y);
              __385.z = (__386.z%__387.z);
              __385.w = (__386.w%__387.w);
            int4 __388;
              int4 __389 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __390 = make_int4(2, 2, 2, 2);
              __388.x = (__389.x/__390.x);
              __388.y = (__389.y/__390.y);
              __388.z = (__389.z/__390.z);
              __388.w = (__389.w/__390.w);
            int4 __391;
            ushort4 __392;
              ushort4 __393;
                ushort4 __394;
                  int4 __395 = make_int4(2, 2, 2, 2);
                  int4 __396 = make_int4(0, 0, 0, 0);
                  __394.x = (__395.x>=__396.x);
                  __394.y = (__395.y>=__396.y);
                  __394.z = (__395.z>=__396.z);
                  __394.w = (__395.w>=__396.w);
                ushort4 __397;
                  int4 __398 = make_int4(0, 0, 0, 0);
                  __397.x = (__385.x>=__398.x);
                  __397.y = (__385.y>=__398.y);
                  __397.z = (__385.z>=__398.z);
                  __397.w = (__385.w>=__398.w);
                __393.x = (__394.x&&__397.x);
                __393.y = (__394.y&&__397.y);
                __393.z = (__394.z&&__397.z);
                __393.w = (__394.w&&__397.w);
              ushort4 __399;
                ushort4 __400;
                  int4 __401 = make_int4(2, 2, 2, 2);
                  int4 __402 = make_int4(0, 0, 0, 0);
                  __400.x = (__401.x<__402.x);
                  __400.y = (__401.y<__402.y);
                  __400.z = (__401.z<__402.z);
                  __400.w = (__401.w<__402.w);
                ushort4 __403;
                  int4 __404 = make_int4(0, 0, 0, 0);
                  __403.x = (__385.x<=__404.x);
                  __403.y = (__385.y<=__404.y);
                  __403.z = (__385.z<=__404.z);
                  __403.w = (__385.w<=__404.w);
                __399.x = (__400.x&&__403.x);
                __399.y = (__400.y&&__403.y);
                __399.z = (__400.z&&__403.z);
                __399.w = (__400.w&&__403.w);
              __392.x = (__393.x||__399.x);
              __392.y = (__393.y||__399.y);
              __392.z = (__393.z||__399.z);
              __392.w = (__393.w||__399.w);
            int4 __405;
              int4 __406 = make_int4(1, 1, 1, 1);
              __405.x = (__388.x-__406.x);
              __405.y = (__388.y-__406.y);
              __405.z = (__388.z-__406.z);
              __405.w = (__388.w-__406.w);
            __391.x = (bool(__392.x)?__388.x:__405.x);
            __391.y = (bool(__392.y)?__388.y:__405.y);
            __391.z = (bool(__392.z)?__388.z:__405.z);
            __391.w = (bool(__392.w)?__388.w:__405.w);
            __383.x = (__384.x+__391.x);
            __383.y = (__384.y+__391.y);
            __383.z = (__384.z+__391.z);
            __383.w = (__384.w+__391.w);
          int __407 = ((0x000000ff << 0) & (B[__383.x] << 0))|((0x000000ff << 8) & (B[__383.y] << 8))|((0x000000ff << 16) & (B[__383.z] << 16))|((0x000000ff << 24) & (B[__383.w] << 24));
          int __408;
          int4 __409;
            int4 __410;
              int4 __411 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __412 = make_int4(2, 2, 2, 2);
              __410.x = (__411.x%__412.x);
              __410.y = (__411.y%__412.y);
              __410.z = (__411.z%__412.z);
              __410.w = (__411.w%__412.w);
            int4 __413;
            ushort4 __414;
              ushort4 __415;
                ushort4 __416;
                  int4 __417 = make_int4(2, 2, 2, 2);
                  int4 __418 = make_int4(0, 0, 0, 0);
                  __416.x = (__417.x>=__418.x);
                  __416.y = (__417.y>=__418.y);
                  __416.z = (__417.z>=__418.z);
                  __416.w = (__417.w>=__418.w);
                ushort4 __419;
                  int4 __420 = make_int4(0, 0, 0, 0);
                  __419.x = (__410.x>=__420.x);
                  __419.y = (__410.y>=__420.y);
                  __419.z = (__410.z>=__420.z);
                  __419.w = (__410.w>=__420.w);
                __415.x = (__416.x&&__419.x);
                __415.y = (__416.y&&__419.y);
                __415.z = (__416.z&&__419.z);
                __415.w = (__416.w&&__419.w);
              ushort4 __421;
                ushort4 __422;
                  int4 __423 = make_int4(2, 2, 2, 2);
                  int4 __424 = make_int4(0, 0, 0, 0);
                  __422.x = (__423.x<__424.x);
                  __422.y = (__423.y<__424.y);
                  __422.z = (__423.z<__424.z);
                  __422.w = (__423.w<__424.w);
                ushort4 __425;
                  int4 __426 = make_int4(0, 0, 0, 0);
                  __425.x = (__410.x<=__426.x);
                  __425.y = (__410.y<=__426.y);
                  __425.z = (__410.z<=__426.z);
                  __425.w = (__410.w<=__426.w);
                __421.x = (__422.x&&__425.x);
                __421.y = (__422.y&&__425.y);
                __421.z = (__422.z&&__425.z);
                __421.w = (__422.w&&__425.w);
              __414.x = (__415.x||__421.x);
              __414.y = (__415.y||__421.y);
              __414.z = (__415.z||__421.z);
              __414.w = (__415.w||__421.w);
            int4 __427;
              int4 __428 = make_int4(2, 2, 2, 2);
              __427.x = (__410.x+__428.x);
              __427.y = (__410.y+__428.y);
              __427.z = (__410.z+__428.z);
              __427.w = (__410.w+__428.w);
            __413.x = (bool(__414.x)?__410.x:__427.x);
            __413.y = (bool(__414.y)?__410.y:__427.y);
            __413.z = (bool(__414.z)?__410.z:__427.z);
            __413.w = (bool(__414.w)?__410.w:__427.w);
            int4 __429 = make_int4(4, 4, 4, 4);
            __409.x = (__413.x*__429.x);
            __409.y = (__413.y*__429.y);
            __409.z = (__413.z*__429.z);
            __409.w = (__413.w*__429.w);
          __408=((signed char)(__409.x) << 0);
          __408=__408 & ~(0x000000ff << 8) |((signed char)(__409.y) << 8);
          __408=__408 & ~(0x000000ff << 16) |((signed char)(__409.z) << 16);
          __408=__408 & ~(0x000000ff << 24) |((signed char)(__409.w) << 24);
          __382=((((char)(__407 >> 0)) >> ((char)(__408 >> 0))) << 0);
          __382=__382 & ~(0x000000ff << 8) |((((char)(__407 >> 8)) >> ((char)(__408 >> 8))) << 8);
          __382=__382 & ~(0x000000ff << 16) |((((char)(__407 >> 16)) >> ((char)(__408 >> 16))) << 16);
          __382=__382 & ~(0x000000ff << 24) |((((char)(__407 >> 24)) >> ((char)(__408 >> 24))) << 24);
        int __430 = (int)252645135;
        __381=((((char)(__382 >> 0)) & ((char)(__430 >> 0))) << 0);
        __381=__381 & ~(0x000000ff << 8) |((((char)(__382 >> 8)) & ((char)(__430 >> 8))) << 8);
        __381=__381 & ~(0x000000ff << 16) |((((char)(__382 >> 16)) & ((char)(__430 >> 16))) << 16);
        __381=__381 & ~(0x000000ff << 24) |((((char)(__382 >> 24)) & ((char)(__430 >> 24))) << 24);
      __380.x = (int)(((char)(__381 >> 0)));
      __380.y = (int)(((char)(__381 >> 8)));
      __380.z = (int)(((char)(__381 >> 16)));
      __380.w = (int)(((char)(__381 >> 24)));
      uint2 __431 = make_uint2(__pack_half2(LUT[__380.x],LUT[__380.y]),__pack_half2(LUT[__380.z],LUT[__380.w]));
      uint2 __432 = make_uint2(__pack_half2(Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 28)], Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 28)]), __pack_half2(Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 28)], Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 28)]));
      ((half2*)(&(__379.x)))->x = (((half2*)(&(__431.x)))->x*((half2*)(&(__432.x)))->x);
      ((half2*)(&(__379.x)))->y = (((half2*)(&(__431.x)))->y*((half2*)(&(__432.x)))->y);
      ((half2*)(&(__379.y)))->x = (((half2*)(&(__431.y)))->x*((half2*)(&(__432.y)))->x);
      ((half2*)(&(__379.y)))->y = (((half2*)(&(__431.y)))->y*((half2*)(&(__432.y)))->y);
    *(uint2*)(B_decode_shared + ((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 4)) + 1120)) = __379;
    __syncthreads();
    for (int k_inner_outer = 0; k_inner_outer < 2; ++k_inner_outer) {
      nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[0], (&(A_shared[(k_inner_outer * 16)])), 40);
      nvcuda::wmma::load_matrix_sync(B_decode_shared_wmma_matrix_b[0], (&(B_decode_shared[(k_inner_outer * 16)])), 40);
      nvcuda::wmma::mma_sync(C_wmma_accumulator[0], A_shared_wmma_matrix_a[0], B_decode_shared_wmma_matrix_b[0], C_wmma_accumulator[0]);
    }
  }
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(C[(((((int)blockIdx.x) / 160) * 40960) + ((((int)blockIdx.x) % 160) * 32))])), C_wmma_accumulator[0], 5120, nvcuda::wmma::mem_row_major);
  __syncthreads();
}



__global__ void __launch_bounds__(32) cutlass_kernel_fp16_nf4_fp16_m16n5120k13824_nt_8x32x32_8x32x16_(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ C) {
            int8_t* B = ((int8_t *)QB);
	 half* LUT = (half *)((int8_t *)QB + 35389440);
	 half* Scales = (half *)((int8_t *)QB + 35389440 + 32);                 
            // const dim3 GridDim(320, 1, 1);
            // const dim3 BlockDim(32, 1, 1);
            // cutlass_kernel_fp16_nf4_fp16_m16n5120k13824_nt_8x32x32_8x32x16_<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
  
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, half> C_wmma_accumulator[1];
  __shared__ half A_shared[320];
  __shared__ half B_decode_shared[1280];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, half, nvcuda::wmma::row_major> A_shared_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, half, nvcuda::wmma::col_major> B_decode_shared_wmma_matrix_b[1];
  nvcuda::wmma::fill_fragment(C_wmma_accumulator[0], __float2half_rn(0.000000e+00f));
  for (int k_outer = 0; k_outer < 432; ++k_outer) {
    __syncthreads();
    *(uint4*)(A_shared + (((((int)threadIdx.x) >> 2) * 40) + ((((int)threadIdx.x) & 3) * 8))) = *(uint4*)(A + (((((((int)blockIdx.x) / 160) * 110592) + ((((int)threadIdx.x) >> 2) * 13824)) + (k_outer * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    uint2 __1;
      int4 __2;
      int __3;
        int __4;
          int4 __5;
            int4 __6 = make_int4((((((((int)blockIdx.x) % 160) * 221184) + ((((int)threadIdx.x) >> 3) * 6912)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)), (((((((int)blockIdx.x) % 160) * 221184) + ((((int)threadIdx.x) >> 3) * 6912)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)), (((((((int)blockIdx.x) % 160) * 221184) + ((((int)threadIdx.x) >> 3) * 6912)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)), (((((((int)blockIdx.x) % 160) * 221184) + ((((int)threadIdx.x) >> 3) * 6912)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)));
            int4 __7;
              int4 __8 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __9 = make_int4(2, 2, 2, 2);
              __7.x = (__8.x%__9.x);
              __7.y = (__8.y%__9.y);
              __7.z = (__8.z%__9.z);
              __7.w = (__8.w%__9.w);
            int4 __10;
              int4 __11 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __12 = make_int4(2, 2, 2, 2);
              __10.x = (__11.x/__12.x);
              __10.y = (__11.y/__12.y);
              __10.z = (__11.z/__12.z);
              __10.w = (__11.w/__12.w);
            int4 __13;
            ushort4 __14;
              ushort4 __15;
                ushort4 __16;
                  int4 __17 = make_int4(2, 2, 2, 2);
                  int4 __18 = make_int4(0, 0, 0, 0);
                  __16.x = (__17.x>=__18.x);
                  __16.y = (__17.y>=__18.y);
                  __16.z = (__17.z>=__18.z);
                  __16.w = (__17.w>=__18.w);
                ushort4 __19;
                  int4 __20 = make_int4(0, 0, 0, 0);
                  __19.x = (__7.x>=__20.x);
                  __19.y = (__7.y>=__20.y);
                  __19.z = (__7.z>=__20.z);
                  __19.w = (__7.w>=__20.w);
                __15.x = (__16.x&&__19.x);
                __15.y = (__16.y&&__19.y);
                __15.z = (__16.z&&__19.z);
                __15.w = (__16.w&&__19.w);
              ushort4 __21;
                ushort4 __22;
                  int4 __23 = make_int4(2, 2, 2, 2);
                  int4 __24 = make_int4(0, 0, 0, 0);
                  __22.x = (__23.x<__24.x);
                  __22.y = (__23.y<__24.y);
                  __22.z = (__23.z<__24.z);
                  __22.w = (__23.w<__24.w);
                ushort4 __25;
                  int4 __26 = make_int4(0, 0, 0, 0);
                  __25.x = (__7.x<=__26.x);
                  __25.y = (__7.y<=__26.y);
                  __25.z = (__7.z<=__26.z);
                  __25.w = (__7.w<=__26.w);
                __21.x = (__22.x&&__25.x);
                __21.y = (__22.y&&__25.y);
                __21.z = (__22.z&&__25.z);
                __21.w = (__22.w&&__25.w);
              __14.x = (__15.x||__21.x);
              __14.y = (__15.y||__21.y);
              __14.z = (__15.z||__21.z);
              __14.w = (__15.w||__21.w);
            int4 __27;
              int4 __28 = make_int4(1, 1, 1, 1);
              __27.x = (__10.x-__28.x);
              __27.y = (__10.y-__28.y);
              __27.z = (__10.z-__28.z);
              __27.w = (__10.w-__28.w);
            __13.x = (bool(__14.x)?__10.x:__27.x);
            __13.y = (bool(__14.y)?__10.y:__27.y);
            __13.z = (bool(__14.z)?__10.z:__27.z);
            __13.w = (bool(__14.w)?__10.w:__27.w);
            __5.x = (__6.x+__13.x);
            __5.y = (__6.y+__13.y);
            __5.z = (__6.z+__13.z);
            __5.w = (__6.w+__13.w);
          int __29 = ((0x000000ff << 0) & (B[__5.x] << 0))|((0x000000ff << 8) & (B[__5.y] << 8))|((0x000000ff << 16) & (B[__5.z] << 16))|((0x000000ff << 24) & (B[__5.w] << 24));
          int __30;
          int4 __31;
            int4 __32;
              int4 __33 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __34 = make_int4(2, 2, 2, 2);
              __32.x = (__33.x%__34.x);
              __32.y = (__33.y%__34.y);
              __32.z = (__33.z%__34.z);
              __32.w = (__33.w%__34.w);
            int4 __35;
            ushort4 __36;
              ushort4 __37;
                ushort4 __38;
                  int4 __39 = make_int4(2, 2, 2, 2);
                  int4 __40 = make_int4(0, 0, 0, 0);
                  __38.x = (__39.x>=__40.x);
                  __38.y = (__39.y>=__40.y);
                  __38.z = (__39.z>=__40.z);
                  __38.w = (__39.w>=__40.w);
                ushort4 __41;
                  int4 __42 = make_int4(0, 0, 0, 0);
                  __41.x = (__32.x>=__42.x);
                  __41.y = (__32.y>=__42.y);
                  __41.z = (__32.z>=__42.z);
                  __41.w = (__32.w>=__42.w);
                __37.x = (__38.x&&__41.x);
                __37.y = (__38.y&&__41.y);
                __37.z = (__38.z&&__41.z);
                __37.w = (__38.w&&__41.w);
              ushort4 __43;
                ushort4 __44;
                  int4 __45 = make_int4(2, 2, 2, 2);
                  int4 __46 = make_int4(0, 0, 0, 0);
                  __44.x = (__45.x<__46.x);
                  __44.y = (__45.y<__46.y);
                  __44.z = (__45.z<__46.z);
                  __44.w = (__45.w<__46.w);
                ushort4 __47;
                  int4 __48 = make_int4(0, 0, 0, 0);
                  __47.x = (__32.x<=__48.x);
                  __47.y = (__32.y<=__48.y);
                  __47.z = (__32.z<=__48.z);
                  __47.w = (__32.w<=__48.w);
                __43.x = (__44.x&&__47.x);
                __43.y = (__44.y&&__47.y);
                __43.z = (__44.z&&__47.z);
                __43.w = (__44.w&&__47.w);
              __36.x = (__37.x||__43.x);
              __36.y = (__37.y||__43.y);
              __36.z = (__37.z||__43.z);
              __36.w = (__37.w||__43.w);
            int4 __49;
              int4 __50 = make_int4(2, 2, 2, 2);
              __49.x = (__32.x+__50.x);
              __49.y = (__32.y+__50.y);
              __49.z = (__32.z+__50.z);
              __49.w = (__32.w+__50.w);
            __35.x = (bool(__36.x)?__32.x:__49.x);
            __35.y = (bool(__36.y)?__32.y:__49.y);
            __35.z = (bool(__36.z)?__32.z:__49.z);
            __35.w = (bool(__36.w)?__32.w:__49.w);
            int4 __51 = make_int4(4, 4, 4, 4);
            __31.x = (__35.x*__51.x);
            __31.y = (__35.y*__51.y);
            __31.z = (__35.z*__51.z);
            __31.w = (__35.w*__51.w);
          __30=((signed char)(__31.x) << 0);
          __30=__30 & ~(0x000000ff << 8) |((signed char)(__31.y) << 8);
          __30=__30 & ~(0x000000ff << 16) |((signed char)(__31.z) << 16);
          __30=__30 & ~(0x000000ff << 24) |((signed char)(__31.w) << 24);
          __4=((((char)(__29 >> 0)) >> ((char)(__30 >> 0))) << 0);
          __4=__4 & ~(0x000000ff << 8) |((((char)(__29 >> 8)) >> ((char)(__30 >> 8))) << 8);
          __4=__4 & ~(0x000000ff << 16) |((((char)(__29 >> 16)) >> ((char)(__30 >> 16))) << 16);
          __4=__4 & ~(0x000000ff << 24) |((((char)(__29 >> 24)) >> ((char)(__30 >> 24))) << 24);
        int __52 = (int)252645135;
        __3=((((char)(__4 >> 0)) & ((char)(__52 >> 0))) << 0);
        __3=__3 & ~(0x000000ff << 8) |((((char)(__4 >> 8)) & ((char)(__52 >> 8))) << 8);
        __3=__3 & ~(0x000000ff << 16) |((((char)(__4 >> 16)) & ((char)(__52 >> 16))) << 16);
        __3=__3 & ~(0x000000ff << 24) |((((char)(__4 >> 24)) & ((char)(__52 >> 24))) << 24);
      __2.x = (int)(((char)(__3 >> 0)));
      __2.y = (int)(((char)(__3 >> 8)));
      __2.z = (int)(((char)(__3 >> 16)));
      __2.w = (int)(((char)(__3 >> 24)));
      uint2 __53 = make_uint2(__pack_half2(LUT[__2.x],LUT[__2.y]),__pack_half2(LUT[__2.z],LUT[__2.w]));
      uint2 __54 = make_uint2(__pack_half2(Scales[((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3))], Scales[((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3))]), __pack_half2(Scales[((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3))], Scales[((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3))]));
      ((half2*)(&(__1.x)))->x = (((half2*)(&(__53.x)))->x*((half2*)(&(__54.x)))->x);
      ((half2*)(&(__1.x)))->y = (((half2*)(&(__53.x)))->y*((half2*)(&(__54.x)))->y);
      ((half2*)(&(__1.y)))->x = (((half2*)(&(__53.y)))->x*((half2*)(&(__54.y)))->x);
      ((half2*)(&(__1.y)))->y = (((half2*)(&(__53.y)))->y*((half2*)(&(__54.y)))->y);
    *(uint2*)(B_decode_shared + (((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 4))) = __1;
    uint2 __55;
      int4 __56;
      int __57;
        int __58;
          int4 __59;
            int4 __60 = make_int4(((((((((int)blockIdx.x) % 160) * 221184) + ((((int)threadIdx.x) >> 3) * 6912)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 27648), ((((((((int)blockIdx.x) % 160) * 221184) + ((((int)threadIdx.x) >> 3) * 6912)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 27648), ((((((((int)blockIdx.x) % 160) * 221184) + ((((int)threadIdx.x) >> 3) * 6912)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 27648), ((((((((int)blockIdx.x) % 160) * 221184) + ((((int)threadIdx.x) >> 3) * 6912)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 27648));
            int4 __61;
              int4 __62 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __63 = make_int4(2, 2, 2, 2);
              __61.x = (__62.x%__63.x);
              __61.y = (__62.y%__63.y);
              __61.z = (__62.z%__63.z);
              __61.w = (__62.w%__63.w);
            int4 __64;
              int4 __65 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __66 = make_int4(2, 2, 2, 2);
              __64.x = (__65.x/__66.x);
              __64.y = (__65.y/__66.y);
              __64.z = (__65.z/__66.z);
              __64.w = (__65.w/__66.w);
            int4 __67;
            ushort4 __68;
              ushort4 __69;
                ushort4 __70;
                  int4 __71 = make_int4(2, 2, 2, 2);
                  int4 __72 = make_int4(0, 0, 0, 0);
                  __70.x = (__71.x>=__72.x);
                  __70.y = (__71.y>=__72.y);
                  __70.z = (__71.z>=__72.z);
                  __70.w = (__71.w>=__72.w);
                ushort4 __73;
                  int4 __74 = make_int4(0, 0, 0, 0);
                  __73.x = (__61.x>=__74.x);
                  __73.y = (__61.y>=__74.y);
                  __73.z = (__61.z>=__74.z);
                  __73.w = (__61.w>=__74.w);
                __69.x = (__70.x&&__73.x);
                __69.y = (__70.y&&__73.y);
                __69.z = (__70.z&&__73.z);
                __69.w = (__70.w&&__73.w);
              ushort4 __75;
                ushort4 __76;
                  int4 __77 = make_int4(2, 2, 2, 2);
                  int4 __78 = make_int4(0, 0, 0, 0);
                  __76.x = (__77.x<__78.x);
                  __76.y = (__77.y<__78.y);
                  __76.z = (__77.z<__78.z);
                  __76.w = (__77.w<__78.w);
                ushort4 __79;
                  int4 __80 = make_int4(0, 0, 0, 0);
                  __79.x = (__61.x<=__80.x);
                  __79.y = (__61.y<=__80.y);
                  __79.z = (__61.z<=__80.z);
                  __79.w = (__61.w<=__80.w);
                __75.x = (__76.x&&__79.x);
                __75.y = (__76.y&&__79.y);
                __75.z = (__76.z&&__79.z);
                __75.w = (__76.w&&__79.w);
              __68.x = (__69.x||__75.x);
              __68.y = (__69.y||__75.y);
              __68.z = (__69.z||__75.z);
              __68.w = (__69.w||__75.w);
            int4 __81;
              int4 __82 = make_int4(1, 1, 1, 1);
              __81.x = (__64.x-__82.x);
              __81.y = (__64.y-__82.y);
              __81.z = (__64.z-__82.z);
              __81.w = (__64.w-__82.w);
            __67.x = (bool(__68.x)?__64.x:__81.x);
            __67.y = (bool(__68.y)?__64.y:__81.y);
            __67.z = (bool(__68.z)?__64.z:__81.z);
            __67.w = (bool(__68.w)?__64.w:__81.w);
            __59.x = (__60.x+__67.x);
            __59.y = (__60.y+__67.y);
            __59.z = (__60.z+__67.z);
            __59.w = (__60.w+__67.w);
          int __83 = ((0x000000ff << 0) & (B[__59.x] << 0))|((0x000000ff << 8) & (B[__59.y] << 8))|((0x000000ff << 16) & (B[__59.z] << 16))|((0x000000ff << 24) & (B[__59.w] << 24));
          int __84;
          int4 __85;
            int4 __86;
              int4 __87 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __88 = make_int4(2, 2, 2, 2);
              __86.x = (__87.x%__88.x);
              __86.y = (__87.y%__88.y);
              __86.z = (__87.z%__88.z);
              __86.w = (__87.w%__88.w);
            int4 __89;
            ushort4 __90;
              ushort4 __91;
                ushort4 __92;
                  int4 __93 = make_int4(2, 2, 2, 2);
                  int4 __94 = make_int4(0, 0, 0, 0);
                  __92.x = (__93.x>=__94.x);
                  __92.y = (__93.y>=__94.y);
                  __92.z = (__93.z>=__94.z);
                  __92.w = (__93.w>=__94.w);
                ushort4 __95;
                  int4 __96 = make_int4(0, 0, 0, 0);
                  __95.x = (__86.x>=__96.x);
                  __95.y = (__86.y>=__96.y);
                  __95.z = (__86.z>=__96.z);
                  __95.w = (__86.w>=__96.w);
                __91.x = (__92.x&&__95.x);
                __91.y = (__92.y&&__95.y);
                __91.z = (__92.z&&__95.z);
                __91.w = (__92.w&&__95.w);
              ushort4 __97;
                ushort4 __98;
                  int4 __99 = make_int4(2, 2, 2, 2);
                  int4 __100 = make_int4(0, 0, 0, 0);
                  __98.x = (__99.x<__100.x);
                  __98.y = (__99.y<__100.y);
                  __98.z = (__99.z<__100.z);
                  __98.w = (__99.w<__100.w);
                ushort4 __101;
                  int4 __102 = make_int4(0, 0, 0, 0);
                  __101.x = (__86.x<=__102.x);
                  __101.y = (__86.y<=__102.y);
                  __101.z = (__86.z<=__102.z);
                  __101.w = (__86.w<=__102.w);
                __97.x = (__98.x&&__101.x);
                __97.y = (__98.y&&__101.y);
                __97.z = (__98.z&&__101.z);
                __97.w = (__98.w&&__101.w);
              __90.x = (__91.x||__97.x);
              __90.y = (__91.y||__97.y);
              __90.z = (__91.z||__97.z);
              __90.w = (__91.w||__97.w);
            int4 __103;
              int4 __104 = make_int4(2, 2, 2, 2);
              __103.x = (__86.x+__104.x);
              __103.y = (__86.y+__104.y);
              __103.z = (__86.z+__104.z);
              __103.w = (__86.w+__104.w);
            __89.x = (bool(__90.x)?__86.x:__103.x);
            __89.y = (bool(__90.y)?__86.y:__103.y);
            __89.z = (bool(__90.z)?__86.z:__103.z);
            __89.w = (bool(__90.w)?__86.w:__103.w);
            int4 __105 = make_int4(4, 4, 4, 4);
            __85.x = (__89.x*__105.x);
            __85.y = (__89.y*__105.y);
            __85.z = (__89.z*__105.z);
            __85.w = (__89.w*__105.w);
          __84=((signed char)(__85.x) << 0);
          __84=__84 & ~(0x000000ff << 8) |((signed char)(__85.y) << 8);
          __84=__84 & ~(0x000000ff << 16) |((signed char)(__85.z) << 16);
          __84=__84 & ~(0x000000ff << 24) |((signed char)(__85.w) << 24);
          __58=((((char)(__83 >> 0)) >> ((char)(__84 >> 0))) << 0);
          __58=__58 & ~(0x000000ff << 8) |((((char)(__83 >> 8)) >> ((char)(__84 >> 8))) << 8);
          __58=__58 & ~(0x000000ff << 16) |((((char)(__83 >> 16)) >> ((char)(__84 >> 16))) << 16);
          __58=__58 & ~(0x000000ff << 24) |((((char)(__83 >> 24)) >> ((char)(__84 >> 24))) << 24);
        int __106 = (int)252645135;
        __57=((((char)(__58 >> 0)) & ((char)(__106 >> 0))) << 0);
        __57=__57 & ~(0x000000ff << 8) |((((char)(__58 >> 8)) & ((char)(__106 >> 8))) << 8);
        __57=__57 & ~(0x000000ff << 16) |((((char)(__58 >> 16)) & ((char)(__106 >> 16))) << 16);
        __57=__57 & ~(0x000000ff << 24) |((((char)(__58 >> 24)) & ((char)(__106 >> 24))) << 24);
      __56.x = (int)(((char)(__57 >> 0)));
      __56.y = (int)(((char)(__57 >> 8)));
      __56.z = (int)(((char)(__57 >> 16)));
      __56.w = (int)(((char)(__57 >> 24)));
      uint2 __107 = make_uint2(__pack_half2(LUT[__56.x],LUT[__56.y]),__pack_half2(LUT[__56.z],LUT[__56.w]));
      uint2 __108 = make_uint2(__pack_half2(Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 4)], Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 4)]), __pack_half2(Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 4)], Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 4)]));
      ((half2*)(&(__55.x)))->x = (((half2*)(&(__107.x)))->x*((half2*)(&(__108.x)))->x);
      ((half2*)(&(__55.x)))->y = (((half2*)(&(__107.x)))->y*((half2*)(&(__108.x)))->y);
      ((half2*)(&(__55.y)))->x = (((half2*)(&(__107.y)))->x*((half2*)(&(__108.y)))->x);
      ((half2*)(&(__55.y)))->y = (((half2*)(&(__107.y)))->y*((half2*)(&(__108.y)))->y);
    *(uint2*)(B_decode_shared + ((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 4)) + 160)) = __55;
    uint2 __109;
      int4 __110;
      int __111;
        int __112;
          int4 __113;
            int4 __114 = make_int4(((((((((int)blockIdx.x) % 160) * 221184) + ((((int)threadIdx.x) >> 3) * 6912)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 55296), ((((((((int)blockIdx.x) % 160) * 221184) + ((((int)threadIdx.x) >> 3) * 6912)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 55296), ((((((((int)blockIdx.x) % 160) * 221184) + ((((int)threadIdx.x) >> 3) * 6912)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 55296), ((((((((int)blockIdx.x) % 160) * 221184) + ((((int)threadIdx.x) >> 3) * 6912)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 55296));
            int4 __115;
              int4 __116 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __117 = make_int4(2, 2, 2, 2);
              __115.x = (__116.x%__117.x);
              __115.y = (__116.y%__117.y);
              __115.z = (__116.z%__117.z);
              __115.w = (__116.w%__117.w);
            int4 __118;
              int4 __119 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __120 = make_int4(2, 2, 2, 2);
              __118.x = (__119.x/__120.x);
              __118.y = (__119.y/__120.y);
              __118.z = (__119.z/__120.z);
              __118.w = (__119.w/__120.w);
            int4 __121;
            ushort4 __122;
              ushort4 __123;
                ushort4 __124;
                  int4 __125 = make_int4(2, 2, 2, 2);
                  int4 __126 = make_int4(0, 0, 0, 0);
                  __124.x = (__125.x>=__126.x);
                  __124.y = (__125.y>=__126.y);
                  __124.z = (__125.z>=__126.z);
                  __124.w = (__125.w>=__126.w);
                ushort4 __127;
                  int4 __128 = make_int4(0, 0, 0, 0);
                  __127.x = (__115.x>=__128.x);
                  __127.y = (__115.y>=__128.y);
                  __127.z = (__115.z>=__128.z);
                  __127.w = (__115.w>=__128.w);
                __123.x = (__124.x&&__127.x);
                __123.y = (__124.y&&__127.y);
                __123.z = (__124.z&&__127.z);
                __123.w = (__124.w&&__127.w);
              ushort4 __129;
                ushort4 __130;
                  int4 __131 = make_int4(2, 2, 2, 2);
                  int4 __132 = make_int4(0, 0, 0, 0);
                  __130.x = (__131.x<__132.x);
                  __130.y = (__131.y<__132.y);
                  __130.z = (__131.z<__132.z);
                  __130.w = (__131.w<__132.w);
                ushort4 __133;
                  int4 __134 = make_int4(0, 0, 0, 0);
                  __133.x = (__115.x<=__134.x);
                  __133.y = (__115.y<=__134.y);
                  __133.z = (__115.z<=__134.z);
                  __133.w = (__115.w<=__134.w);
                __129.x = (__130.x&&__133.x);
                __129.y = (__130.y&&__133.y);
                __129.z = (__130.z&&__133.z);
                __129.w = (__130.w&&__133.w);
              __122.x = (__123.x||__129.x);
              __122.y = (__123.y||__129.y);
              __122.z = (__123.z||__129.z);
              __122.w = (__123.w||__129.w);
            int4 __135;
              int4 __136 = make_int4(1, 1, 1, 1);
              __135.x = (__118.x-__136.x);
              __135.y = (__118.y-__136.y);
              __135.z = (__118.z-__136.z);
              __135.w = (__118.w-__136.w);
            __121.x = (bool(__122.x)?__118.x:__135.x);
            __121.y = (bool(__122.y)?__118.y:__135.y);
            __121.z = (bool(__122.z)?__118.z:__135.z);
            __121.w = (bool(__122.w)?__118.w:__135.w);
            __113.x = (__114.x+__121.x);
            __113.y = (__114.y+__121.y);
            __113.z = (__114.z+__121.z);
            __113.w = (__114.w+__121.w);
          int __137 = ((0x000000ff << 0) & (B[__113.x] << 0))|((0x000000ff << 8) & (B[__113.y] << 8))|((0x000000ff << 16) & (B[__113.z] << 16))|((0x000000ff << 24) & (B[__113.w] << 24));
          int __138;
          int4 __139;
            int4 __140;
              int4 __141 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __142 = make_int4(2, 2, 2, 2);
              __140.x = (__141.x%__142.x);
              __140.y = (__141.y%__142.y);
              __140.z = (__141.z%__142.z);
              __140.w = (__141.w%__142.w);
            int4 __143;
            ushort4 __144;
              ushort4 __145;
                ushort4 __146;
                  int4 __147 = make_int4(2, 2, 2, 2);
                  int4 __148 = make_int4(0, 0, 0, 0);
                  __146.x = (__147.x>=__148.x);
                  __146.y = (__147.y>=__148.y);
                  __146.z = (__147.z>=__148.z);
                  __146.w = (__147.w>=__148.w);
                ushort4 __149;
                  int4 __150 = make_int4(0, 0, 0, 0);
                  __149.x = (__140.x>=__150.x);
                  __149.y = (__140.y>=__150.y);
                  __149.z = (__140.z>=__150.z);
                  __149.w = (__140.w>=__150.w);
                __145.x = (__146.x&&__149.x);
                __145.y = (__146.y&&__149.y);
                __145.z = (__146.z&&__149.z);
                __145.w = (__146.w&&__149.w);
              ushort4 __151;
                ushort4 __152;
                  int4 __153 = make_int4(2, 2, 2, 2);
                  int4 __154 = make_int4(0, 0, 0, 0);
                  __152.x = (__153.x<__154.x);
                  __152.y = (__153.y<__154.y);
                  __152.z = (__153.z<__154.z);
                  __152.w = (__153.w<__154.w);
                ushort4 __155;
                  int4 __156 = make_int4(0, 0, 0, 0);
                  __155.x = (__140.x<=__156.x);
                  __155.y = (__140.y<=__156.y);
                  __155.z = (__140.z<=__156.z);
                  __155.w = (__140.w<=__156.w);
                __151.x = (__152.x&&__155.x);
                __151.y = (__152.y&&__155.y);
                __151.z = (__152.z&&__155.z);
                __151.w = (__152.w&&__155.w);
              __144.x = (__145.x||__151.x);
              __144.y = (__145.y||__151.y);
              __144.z = (__145.z||__151.z);
              __144.w = (__145.w||__151.w);
            int4 __157;
              int4 __158 = make_int4(2, 2, 2, 2);
              __157.x = (__140.x+__158.x);
              __157.y = (__140.y+__158.y);
              __157.z = (__140.z+__158.z);
              __157.w = (__140.w+__158.w);
            __143.x = (bool(__144.x)?__140.x:__157.x);
            __143.y = (bool(__144.y)?__140.y:__157.y);
            __143.z = (bool(__144.z)?__140.z:__157.z);
            __143.w = (bool(__144.w)?__140.w:__157.w);
            int4 __159 = make_int4(4, 4, 4, 4);
            __139.x = (__143.x*__159.x);
            __139.y = (__143.y*__159.y);
            __139.z = (__143.z*__159.z);
            __139.w = (__143.w*__159.w);
          __138=((signed char)(__139.x) << 0);
          __138=__138 & ~(0x000000ff << 8) |((signed char)(__139.y) << 8);
          __138=__138 & ~(0x000000ff << 16) |((signed char)(__139.z) << 16);
          __138=__138 & ~(0x000000ff << 24) |((signed char)(__139.w) << 24);
          __112=((((char)(__137 >> 0)) >> ((char)(__138 >> 0))) << 0);
          __112=__112 & ~(0x000000ff << 8) |((((char)(__137 >> 8)) >> ((char)(__138 >> 8))) << 8);
          __112=__112 & ~(0x000000ff << 16) |((((char)(__137 >> 16)) >> ((char)(__138 >> 16))) << 16);
          __112=__112 & ~(0x000000ff << 24) |((((char)(__137 >> 24)) >> ((char)(__138 >> 24))) << 24);
        int __160 = (int)252645135;
        __111=((((char)(__112 >> 0)) & ((char)(__160 >> 0))) << 0);
        __111=__111 & ~(0x000000ff << 8) |((((char)(__112 >> 8)) & ((char)(__160 >> 8))) << 8);
        __111=__111 & ~(0x000000ff << 16) |((((char)(__112 >> 16)) & ((char)(__160 >> 16))) << 16);
        __111=__111 & ~(0x000000ff << 24) |((((char)(__112 >> 24)) & ((char)(__160 >> 24))) << 24);
      __110.x = (int)(((char)(__111 >> 0)));
      __110.y = (int)(((char)(__111 >> 8)));
      __110.z = (int)(((char)(__111 >> 16)));
      __110.w = (int)(((char)(__111 >> 24)));
      uint2 __161 = make_uint2(__pack_half2(LUT[__110.x],LUT[__110.y]),__pack_half2(LUT[__110.z],LUT[__110.w]));
      uint2 __162 = make_uint2(__pack_half2(Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 8)], Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 8)]), __pack_half2(Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 8)], Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 8)]));
      ((half2*)(&(__109.x)))->x = (((half2*)(&(__161.x)))->x*((half2*)(&(__162.x)))->x);
      ((half2*)(&(__109.x)))->y = (((half2*)(&(__161.x)))->y*((half2*)(&(__162.x)))->y);
      ((half2*)(&(__109.y)))->x = (((half2*)(&(__161.y)))->x*((half2*)(&(__162.y)))->x);
      ((half2*)(&(__109.y)))->y = (((half2*)(&(__161.y)))->y*((half2*)(&(__162.y)))->y);
    *(uint2*)(B_decode_shared + ((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 4)) + 320)) = __109;
    uint2 __163;
      int4 __164;
      int __165;
        int __166;
          int4 __167;
            int4 __168 = make_int4(((((((((int)blockIdx.x) % 160) * 221184) + ((((int)threadIdx.x) >> 3) * 6912)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 82944), ((((((((int)blockIdx.x) % 160) * 221184) + ((((int)threadIdx.x) >> 3) * 6912)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 82944), ((((((((int)blockIdx.x) % 160) * 221184) + ((((int)threadIdx.x) >> 3) * 6912)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 82944), ((((((((int)blockIdx.x) % 160) * 221184) + ((((int)threadIdx.x) >> 3) * 6912)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 82944));
            int4 __169;
              int4 __170 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __171 = make_int4(2, 2, 2, 2);
              __169.x = (__170.x%__171.x);
              __169.y = (__170.y%__171.y);
              __169.z = (__170.z%__171.z);
              __169.w = (__170.w%__171.w);
            int4 __172;
              int4 __173 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __174 = make_int4(2, 2, 2, 2);
              __172.x = (__173.x/__174.x);
              __172.y = (__173.y/__174.y);
              __172.z = (__173.z/__174.z);
              __172.w = (__173.w/__174.w);
            int4 __175;
            ushort4 __176;
              ushort4 __177;
                ushort4 __178;
                  int4 __179 = make_int4(2, 2, 2, 2);
                  int4 __180 = make_int4(0, 0, 0, 0);
                  __178.x = (__179.x>=__180.x);
                  __178.y = (__179.y>=__180.y);
                  __178.z = (__179.z>=__180.z);
                  __178.w = (__179.w>=__180.w);
                ushort4 __181;
                  int4 __182 = make_int4(0, 0, 0, 0);
                  __181.x = (__169.x>=__182.x);
                  __181.y = (__169.y>=__182.y);
                  __181.z = (__169.z>=__182.z);
                  __181.w = (__169.w>=__182.w);
                __177.x = (__178.x&&__181.x);
                __177.y = (__178.y&&__181.y);
                __177.z = (__178.z&&__181.z);
                __177.w = (__178.w&&__181.w);
              ushort4 __183;
                ushort4 __184;
                  int4 __185 = make_int4(2, 2, 2, 2);
                  int4 __186 = make_int4(0, 0, 0, 0);
                  __184.x = (__185.x<__186.x);
                  __184.y = (__185.y<__186.y);
                  __184.z = (__185.z<__186.z);
                  __184.w = (__185.w<__186.w);
                ushort4 __187;
                  int4 __188 = make_int4(0, 0, 0, 0);
                  __187.x = (__169.x<=__188.x);
                  __187.y = (__169.y<=__188.y);
                  __187.z = (__169.z<=__188.z);
                  __187.w = (__169.w<=__188.w);
                __183.x = (__184.x&&__187.x);
                __183.y = (__184.y&&__187.y);
                __183.z = (__184.z&&__187.z);
                __183.w = (__184.w&&__187.w);
              __176.x = (__177.x||__183.x);
              __176.y = (__177.y||__183.y);
              __176.z = (__177.z||__183.z);
              __176.w = (__177.w||__183.w);
            int4 __189;
              int4 __190 = make_int4(1, 1, 1, 1);
              __189.x = (__172.x-__190.x);
              __189.y = (__172.y-__190.y);
              __189.z = (__172.z-__190.z);
              __189.w = (__172.w-__190.w);
            __175.x = (bool(__176.x)?__172.x:__189.x);
            __175.y = (bool(__176.y)?__172.y:__189.y);
            __175.z = (bool(__176.z)?__172.z:__189.z);
            __175.w = (bool(__176.w)?__172.w:__189.w);
            __167.x = (__168.x+__175.x);
            __167.y = (__168.y+__175.y);
            __167.z = (__168.z+__175.z);
            __167.w = (__168.w+__175.w);
          int __191 = ((0x000000ff << 0) & (B[__167.x] << 0))|((0x000000ff << 8) & (B[__167.y] << 8))|((0x000000ff << 16) & (B[__167.z] << 16))|((0x000000ff << 24) & (B[__167.w] << 24));
          int __192;
          int4 __193;
            int4 __194;
              int4 __195 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __196 = make_int4(2, 2, 2, 2);
              __194.x = (__195.x%__196.x);
              __194.y = (__195.y%__196.y);
              __194.z = (__195.z%__196.z);
              __194.w = (__195.w%__196.w);
            int4 __197;
            ushort4 __198;
              ushort4 __199;
                ushort4 __200;
                  int4 __201 = make_int4(2, 2, 2, 2);
                  int4 __202 = make_int4(0, 0, 0, 0);
                  __200.x = (__201.x>=__202.x);
                  __200.y = (__201.y>=__202.y);
                  __200.z = (__201.z>=__202.z);
                  __200.w = (__201.w>=__202.w);
                ushort4 __203;
                  int4 __204 = make_int4(0, 0, 0, 0);
                  __203.x = (__194.x>=__204.x);
                  __203.y = (__194.y>=__204.y);
                  __203.z = (__194.z>=__204.z);
                  __203.w = (__194.w>=__204.w);
                __199.x = (__200.x&&__203.x);
                __199.y = (__200.y&&__203.y);
                __199.z = (__200.z&&__203.z);
                __199.w = (__200.w&&__203.w);
              ushort4 __205;
                ushort4 __206;
                  int4 __207 = make_int4(2, 2, 2, 2);
                  int4 __208 = make_int4(0, 0, 0, 0);
                  __206.x = (__207.x<__208.x);
                  __206.y = (__207.y<__208.y);
                  __206.z = (__207.z<__208.z);
                  __206.w = (__207.w<__208.w);
                ushort4 __209;
                  int4 __210 = make_int4(0, 0, 0, 0);
                  __209.x = (__194.x<=__210.x);
                  __209.y = (__194.y<=__210.y);
                  __209.z = (__194.z<=__210.z);
                  __209.w = (__194.w<=__210.w);
                __205.x = (__206.x&&__209.x);
                __205.y = (__206.y&&__209.y);
                __205.z = (__206.z&&__209.z);
                __205.w = (__206.w&&__209.w);
              __198.x = (__199.x||__205.x);
              __198.y = (__199.y||__205.y);
              __198.z = (__199.z||__205.z);
              __198.w = (__199.w||__205.w);
            int4 __211;
              int4 __212 = make_int4(2, 2, 2, 2);
              __211.x = (__194.x+__212.x);
              __211.y = (__194.y+__212.y);
              __211.z = (__194.z+__212.z);
              __211.w = (__194.w+__212.w);
            __197.x = (bool(__198.x)?__194.x:__211.x);
            __197.y = (bool(__198.y)?__194.y:__211.y);
            __197.z = (bool(__198.z)?__194.z:__211.z);
            __197.w = (bool(__198.w)?__194.w:__211.w);
            int4 __213 = make_int4(4, 4, 4, 4);
            __193.x = (__197.x*__213.x);
            __193.y = (__197.y*__213.y);
            __193.z = (__197.z*__213.z);
            __193.w = (__197.w*__213.w);
          __192=((signed char)(__193.x) << 0);
          __192=__192 & ~(0x000000ff << 8) |((signed char)(__193.y) << 8);
          __192=__192 & ~(0x000000ff << 16) |((signed char)(__193.z) << 16);
          __192=__192 & ~(0x000000ff << 24) |((signed char)(__193.w) << 24);
          __166=((((char)(__191 >> 0)) >> ((char)(__192 >> 0))) << 0);
          __166=__166 & ~(0x000000ff << 8) |((((char)(__191 >> 8)) >> ((char)(__192 >> 8))) << 8);
          __166=__166 & ~(0x000000ff << 16) |((((char)(__191 >> 16)) >> ((char)(__192 >> 16))) << 16);
          __166=__166 & ~(0x000000ff << 24) |((((char)(__191 >> 24)) >> ((char)(__192 >> 24))) << 24);
        int __214 = (int)252645135;
        __165=((((char)(__166 >> 0)) & ((char)(__214 >> 0))) << 0);
        __165=__165 & ~(0x000000ff << 8) |((((char)(__166 >> 8)) & ((char)(__214 >> 8))) << 8);
        __165=__165 & ~(0x000000ff << 16) |((((char)(__166 >> 16)) & ((char)(__214 >> 16))) << 16);
        __165=__165 & ~(0x000000ff << 24) |((((char)(__166 >> 24)) & ((char)(__214 >> 24))) << 24);
      __164.x = (int)(((char)(__165 >> 0)));
      __164.y = (int)(((char)(__165 >> 8)));
      __164.z = (int)(((char)(__165 >> 16)));
      __164.w = (int)(((char)(__165 >> 24)));
      uint2 __215 = make_uint2(__pack_half2(LUT[__164.x],LUT[__164.y]),__pack_half2(LUT[__164.z],LUT[__164.w]));
      uint2 __216 = make_uint2(__pack_half2(Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 12)], Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 12)]), __pack_half2(Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 12)], Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 12)]));
      ((half2*)(&(__163.x)))->x = (((half2*)(&(__215.x)))->x*((half2*)(&(__216.x)))->x);
      ((half2*)(&(__163.x)))->y = (((half2*)(&(__215.x)))->y*((half2*)(&(__216.x)))->y);
      ((half2*)(&(__163.y)))->x = (((half2*)(&(__215.y)))->x*((half2*)(&(__216.y)))->x);
      ((half2*)(&(__163.y)))->y = (((half2*)(&(__215.y)))->y*((half2*)(&(__216.y)))->y);
    *(uint2*)(B_decode_shared + ((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 4)) + 480)) = __163;
    uint2 __217;
      int4 __218;
      int __219;
        int __220;
          int4 __221;
            int4 __222 = make_int4(((((((((int)blockIdx.x) % 160) * 221184) + ((((int)threadIdx.x) >> 3) * 6912)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 110592), ((((((((int)blockIdx.x) % 160) * 221184) + ((((int)threadIdx.x) >> 3) * 6912)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 110592), ((((((((int)blockIdx.x) % 160) * 221184) + ((((int)threadIdx.x) >> 3) * 6912)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 110592), ((((((((int)blockIdx.x) % 160) * 221184) + ((((int)threadIdx.x) >> 3) * 6912)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 110592));
            int4 __223;
              int4 __224 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __225 = make_int4(2, 2, 2, 2);
              __223.x = (__224.x%__225.x);
              __223.y = (__224.y%__225.y);
              __223.z = (__224.z%__225.z);
              __223.w = (__224.w%__225.w);
            int4 __226;
              int4 __227 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __228 = make_int4(2, 2, 2, 2);
              __226.x = (__227.x/__228.x);
              __226.y = (__227.y/__228.y);
              __226.z = (__227.z/__228.z);
              __226.w = (__227.w/__228.w);
            int4 __229;
            ushort4 __230;
              ushort4 __231;
                ushort4 __232;
                  int4 __233 = make_int4(2, 2, 2, 2);
                  int4 __234 = make_int4(0, 0, 0, 0);
                  __232.x = (__233.x>=__234.x);
                  __232.y = (__233.y>=__234.y);
                  __232.z = (__233.z>=__234.z);
                  __232.w = (__233.w>=__234.w);
                ushort4 __235;
                  int4 __236 = make_int4(0, 0, 0, 0);
                  __235.x = (__223.x>=__236.x);
                  __235.y = (__223.y>=__236.y);
                  __235.z = (__223.z>=__236.z);
                  __235.w = (__223.w>=__236.w);
                __231.x = (__232.x&&__235.x);
                __231.y = (__232.y&&__235.y);
                __231.z = (__232.z&&__235.z);
                __231.w = (__232.w&&__235.w);
              ushort4 __237;
                ushort4 __238;
                  int4 __239 = make_int4(2, 2, 2, 2);
                  int4 __240 = make_int4(0, 0, 0, 0);
                  __238.x = (__239.x<__240.x);
                  __238.y = (__239.y<__240.y);
                  __238.z = (__239.z<__240.z);
                  __238.w = (__239.w<__240.w);
                ushort4 __241;
                  int4 __242 = make_int4(0, 0, 0, 0);
                  __241.x = (__223.x<=__242.x);
                  __241.y = (__223.y<=__242.y);
                  __241.z = (__223.z<=__242.z);
                  __241.w = (__223.w<=__242.w);
                __237.x = (__238.x&&__241.x);
                __237.y = (__238.y&&__241.y);
                __237.z = (__238.z&&__241.z);
                __237.w = (__238.w&&__241.w);
              __230.x = (__231.x||__237.x);
              __230.y = (__231.y||__237.y);
              __230.z = (__231.z||__237.z);
              __230.w = (__231.w||__237.w);
            int4 __243;
              int4 __244 = make_int4(1, 1, 1, 1);
              __243.x = (__226.x-__244.x);
              __243.y = (__226.y-__244.y);
              __243.z = (__226.z-__244.z);
              __243.w = (__226.w-__244.w);
            __229.x = (bool(__230.x)?__226.x:__243.x);
            __229.y = (bool(__230.y)?__226.y:__243.y);
            __229.z = (bool(__230.z)?__226.z:__243.z);
            __229.w = (bool(__230.w)?__226.w:__243.w);
            __221.x = (__222.x+__229.x);
            __221.y = (__222.y+__229.y);
            __221.z = (__222.z+__229.z);
            __221.w = (__222.w+__229.w);
          int __245 = ((0x000000ff << 0) & (B[__221.x] << 0))|((0x000000ff << 8) & (B[__221.y] << 8))|((0x000000ff << 16) & (B[__221.z] << 16))|((0x000000ff << 24) & (B[__221.w] << 24));
          int __246;
          int4 __247;
            int4 __248;
              int4 __249 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __250 = make_int4(2, 2, 2, 2);
              __248.x = (__249.x%__250.x);
              __248.y = (__249.y%__250.y);
              __248.z = (__249.z%__250.z);
              __248.w = (__249.w%__250.w);
            int4 __251;
            ushort4 __252;
              ushort4 __253;
                ushort4 __254;
                  int4 __255 = make_int4(2, 2, 2, 2);
                  int4 __256 = make_int4(0, 0, 0, 0);
                  __254.x = (__255.x>=__256.x);
                  __254.y = (__255.y>=__256.y);
                  __254.z = (__255.z>=__256.z);
                  __254.w = (__255.w>=__256.w);
                ushort4 __257;
                  int4 __258 = make_int4(0, 0, 0, 0);
                  __257.x = (__248.x>=__258.x);
                  __257.y = (__248.y>=__258.y);
                  __257.z = (__248.z>=__258.z);
                  __257.w = (__248.w>=__258.w);
                __253.x = (__254.x&&__257.x);
                __253.y = (__254.y&&__257.y);
                __253.z = (__254.z&&__257.z);
                __253.w = (__254.w&&__257.w);
              ushort4 __259;
                ushort4 __260;
                  int4 __261 = make_int4(2, 2, 2, 2);
                  int4 __262 = make_int4(0, 0, 0, 0);
                  __260.x = (__261.x<__262.x);
                  __260.y = (__261.y<__262.y);
                  __260.z = (__261.z<__262.z);
                  __260.w = (__261.w<__262.w);
                ushort4 __263;
                  int4 __264 = make_int4(0, 0, 0, 0);
                  __263.x = (__248.x<=__264.x);
                  __263.y = (__248.y<=__264.y);
                  __263.z = (__248.z<=__264.z);
                  __263.w = (__248.w<=__264.w);
                __259.x = (__260.x&&__263.x);
                __259.y = (__260.y&&__263.y);
                __259.z = (__260.z&&__263.z);
                __259.w = (__260.w&&__263.w);
              __252.x = (__253.x||__259.x);
              __252.y = (__253.y||__259.y);
              __252.z = (__253.z||__259.z);
              __252.w = (__253.w||__259.w);
            int4 __265;
              int4 __266 = make_int4(2, 2, 2, 2);
              __265.x = (__248.x+__266.x);
              __265.y = (__248.y+__266.y);
              __265.z = (__248.z+__266.z);
              __265.w = (__248.w+__266.w);
            __251.x = (bool(__252.x)?__248.x:__265.x);
            __251.y = (bool(__252.y)?__248.y:__265.y);
            __251.z = (bool(__252.z)?__248.z:__265.z);
            __251.w = (bool(__252.w)?__248.w:__265.w);
            int4 __267 = make_int4(4, 4, 4, 4);
            __247.x = (__251.x*__267.x);
            __247.y = (__251.y*__267.y);
            __247.z = (__251.z*__267.z);
            __247.w = (__251.w*__267.w);
          __246=((signed char)(__247.x) << 0);
          __246=__246 & ~(0x000000ff << 8) |((signed char)(__247.y) << 8);
          __246=__246 & ~(0x000000ff << 16) |((signed char)(__247.z) << 16);
          __246=__246 & ~(0x000000ff << 24) |((signed char)(__247.w) << 24);
          __220=((((char)(__245 >> 0)) >> ((char)(__246 >> 0))) << 0);
          __220=__220 & ~(0x000000ff << 8) |((((char)(__245 >> 8)) >> ((char)(__246 >> 8))) << 8);
          __220=__220 & ~(0x000000ff << 16) |((((char)(__245 >> 16)) >> ((char)(__246 >> 16))) << 16);
          __220=__220 & ~(0x000000ff << 24) |((((char)(__245 >> 24)) >> ((char)(__246 >> 24))) << 24);
        int __268 = (int)252645135;
        __219=((((char)(__220 >> 0)) & ((char)(__268 >> 0))) << 0);
        __219=__219 & ~(0x000000ff << 8) |((((char)(__220 >> 8)) & ((char)(__268 >> 8))) << 8);
        __219=__219 & ~(0x000000ff << 16) |((((char)(__220 >> 16)) & ((char)(__268 >> 16))) << 16);
        __219=__219 & ~(0x000000ff << 24) |((((char)(__220 >> 24)) & ((char)(__268 >> 24))) << 24);
      __218.x = (int)(((char)(__219 >> 0)));
      __218.y = (int)(((char)(__219 >> 8)));
      __218.z = (int)(((char)(__219 >> 16)));
      __218.w = (int)(((char)(__219 >> 24)));
      uint2 __269 = make_uint2(__pack_half2(LUT[__218.x],LUT[__218.y]),__pack_half2(LUT[__218.z],LUT[__218.w]));
      uint2 __270 = make_uint2(__pack_half2(Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 16)], Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 16)]), __pack_half2(Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 16)], Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 16)]));
      ((half2*)(&(__217.x)))->x = (((half2*)(&(__269.x)))->x*((half2*)(&(__270.x)))->x);
      ((half2*)(&(__217.x)))->y = (((half2*)(&(__269.x)))->y*((half2*)(&(__270.x)))->y);
      ((half2*)(&(__217.y)))->x = (((half2*)(&(__269.y)))->x*((half2*)(&(__270.y)))->x);
      ((half2*)(&(__217.y)))->y = (((half2*)(&(__269.y)))->y*((half2*)(&(__270.y)))->y);
    *(uint2*)(B_decode_shared + ((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 4)) + 640)) = __217;
    uint2 __271;
      int4 __272;
      int __273;
        int __274;
          int4 __275;
            int4 __276 = make_int4(((((((((int)blockIdx.x) % 160) * 221184) + ((((int)threadIdx.x) >> 3) * 6912)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 138240), ((((((((int)blockIdx.x) % 160) * 221184) + ((((int)threadIdx.x) >> 3) * 6912)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 138240), ((((((((int)blockIdx.x) % 160) * 221184) + ((((int)threadIdx.x) >> 3) * 6912)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 138240), ((((((((int)blockIdx.x) % 160) * 221184) + ((((int)threadIdx.x) >> 3) * 6912)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 138240));
            int4 __277;
              int4 __278 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __279 = make_int4(2, 2, 2, 2);
              __277.x = (__278.x%__279.x);
              __277.y = (__278.y%__279.y);
              __277.z = (__278.z%__279.z);
              __277.w = (__278.w%__279.w);
            int4 __280;
              int4 __281 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __282 = make_int4(2, 2, 2, 2);
              __280.x = (__281.x/__282.x);
              __280.y = (__281.y/__282.y);
              __280.z = (__281.z/__282.z);
              __280.w = (__281.w/__282.w);
            int4 __283;
            ushort4 __284;
              ushort4 __285;
                ushort4 __286;
                  int4 __287 = make_int4(2, 2, 2, 2);
                  int4 __288 = make_int4(0, 0, 0, 0);
                  __286.x = (__287.x>=__288.x);
                  __286.y = (__287.y>=__288.y);
                  __286.z = (__287.z>=__288.z);
                  __286.w = (__287.w>=__288.w);
                ushort4 __289;
                  int4 __290 = make_int4(0, 0, 0, 0);
                  __289.x = (__277.x>=__290.x);
                  __289.y = (__277.y>=__290.y);
                  __289.z = (__277.z>=__290.z);
                  __289.w = (__277.w>=__290.w);
                __285.x = (__286.x&&__289.x);
                __285.y = (__286.y&&__289.y);
                __285.z = (__286.z&&__289.z);
                __285.w = (__286.w&&__289.w);
              ushort4 __291;
                ushort4 __292;
                  int4 __293 = make_int4(2, 2, 2, 2);
                  int4 __294 = make_int4(0, 0, 0, 0);
                  __292.x = (__293.x<__294.x);
                  __292.y = (__293.y<__294.y);
                  __292.z = (__293.z<__294.z);
                  __292.w = (__293.w<__294.w);
                ushort4 __295;
                  int4 __296 = make_int4(0, 0, 0, 0);
                  __295.x = (__277.x<=__296.x);
                  __295.y = (__277.y<=__296.y);
                  __295.z = (__277.z<=__296.z);
                  __295.w = (__277.w<=__296.w);
                __291.x = (__292.x&&__295.x);
                __291.y = (__292.y&&__295.y);
                __291.z = (__292.z&&__295.z);
                __291.w = (__292.w&&__295.w);
              __284.x = (__285.x||__291.x);
              __284.y = (__285.y||__291.y);
              __284.z = (__285.z||__291.z);
              __284.w = (__285.w||__291.w);
            int4 __297;
              int4 __298 = make_int4(1, 1, 1, 1);
              __297.x = (__280.x-__298.x);
              __297.y = (__280.y-__298.y);
              __297.z = (__280.z-__298.z);
              __297.w = (__280.w-__298.w);
            __283.x = (bool(__284.x)?__280.x:__297.x);
            __283.y = (bool(__284.y)?__280.y:__297.y);
            __283.z = (bool(__284.z)?__280.z:__297.z);
            __283.w = (bool(__284.w)?__280.w:__297.w);
            __275.x = (__276.x+__283.x);
            __275.y = (__276.y+__283.y);
            __275.z = (__276.z+__283.z);
            __275.w = (__276.w+__283.w);
          int __299 = ((0x000000ff << 0) & (B[__275.x] << 0))|((0x000000ff << 8) & (B[__275.y] << 8))|((0x000000ff << 16) & (B[__275.z] << 16))|((0x000000ff << 24) & (B[__275.w] << 24));
          int __300;
          int4 __301;
            int4 __302;
              int4 __303 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __304 = make_int4(2, 2, 2, 2);
              __302.x = (__303.x%__304.x);
              __302.y = (__303.y%__304.y);
              __302.z = (__303.z%__304.z);
              __302.w = (__303.w%__304.w);
            int4 __305;
            ushort4 __306;
              ushort4 __307;
                ushort4 __308;
                  int4 __309 = make_int4(2, 2, 2, 2);
                  int4 __310 = make_int4(0, 0, 0, 0);
                  __308.x = (__309.x>=__310.x);
                  __308.y = (__309.y>=__310.y);
                  __308.z = (__309.z>=__310.z);
                  __308.w = (__309.w>=__310.w);
                ushort4 __311;
                  int4 __312 = make_int4(0, 0, 0, 0);
                  __311.x = (__302.x>=__312.x);
                  __311.y = (__302.y>=__312.y);
                  __311.z = (__302.z>=__312.z);
                  __311.w = (__302.w>=__312.w);
                __307.x = (__308.x&&__311.x);
                __307.y = (__308.y&&__311.y);
                __307.z = (__308.z&&__311.z);
                __307.w = (__308.w&&__311.w);
              ushort4 __313;
                ushort4 __314;
                  int4 __315 = make_int4(2, 2, 2, 2);
                  int4 __316 = make_int4(0, 0, 0, 0);
                  __314.x = (__315.x<__316.x);
                  __314.y = (__315.y<__316.y);
                  __314.z = (__315.z<__316.z);
                  __314.w = (__315.w<__316.w);
                ushort4 __317;
                  int4 __318 = make_int4(0, 0, 0, 0);
                  __317.x = (__302.x<=__318.x);
                  __317.y = (__302.y<=__318.y);
                  __317.z = (__302.z<=__318.z);
                  __317.w = (__302.w<=__318.w);
                __313.x = (__314.x&&__317.x);
                __313.y = (__314.y&&__317.y);
                __313.z = (__314.z&&__317.z);
                __313.w = (__314.w&&__317.w);
              __306.x = (__307.x||__313.x);
              __306.y = (__307.y||__313.y);
              __306.z = (__307.z||__313.z);
              __306.w = (__307.w||__313.w);
            int4 __319;
              int4 __320 = make_int4(2, 2, 2, 2);
              __319.x = (__302.x+__320.x);
              __319.y = (__302.y+__320.y);
              __319.z = (__302.z+__320.z);
              __319.w = (__302.w+__320.w);
            __305.x = (bool(__306.x)?__302.x:__319.x);
            __305.y = (bool(__306.y)?__302.y:__319.y);
            __305.z = (bool(__306.z)?__302.z:__319.z);
            __305.w = (bool(__306.w)?__302.w:__319.w);
            int4 __321 = make_int4(4, 4, 4, 4);
            __301.x = (__305.x*__321.x);
            __301.y = (__305.y*__321.y);
            __301.z = (__305.z*__321.z);
            __301.w = (__305.w*__321.w);
          __300=((signed char)(__301.x) << 0);
          __300=__300 & ~(0x000000ff << 8) |((signed char)(__301.y) << 8);
          __300=__300 & ~(0x000000ff << 16) |((signed char)(__301.z) << 16);
          __300=__300 & ~(0x000000ff << 24) |((signed char)(__301.w) << 24);
          __274=((((char)(__299 >> 0)) >> ((char)(__300 >> 0))) << 0);
          __274=__274 & ~(0x000000ff << 8) |((((char)(__299 >> 8)) >> ((char)(__300 >> 8))) << 8);
          __274=__274 & ~(0x000000ff << 16) |((((char)(__299 >> 16)) >> ((char)(__300 >> 16))) << 16);
          __274=__274 & ~(0x000000ff << 24) |((((char)(__299 >> 24)) >> ((char)(__300 >> 24))) << 24);
        int __322 = (int)252645135;
        __273=((((char)(__274 >> 0)) & ((char)(__322 >> 0))) << 0);
        __273=__273 & ~(0x000000ff << 8) |((((char)(__274 >> 8)) & ((char)(__322 >> 8))) << 8);
        __273=__273 & ~(0x000000ff << 16) |((((char)(__274 >> 16)) & ((char)(__322 >> 16))) << 16);
        __273=__273 & ~(0x000000ff << 24) |((((char)(__274 >> 24)) & ((char)(__322 >> 24))) << 24);
      __272.x = (int)(((char)(__273 >> 0)));
      __272.y = (int)(((char)(__273 >> 8)));
      __272.z = (int)(((char)(__273 >> 16)));
      __272.w = (int)(((char)(__273 >> 24)));
      uint2 __323 = make_uint2(__pack_half2(LUT[__272.x],LUT[__272.y]),__pack_half2(LUT[__272.z],LUT[__272.w]));
      uint2 __324 = make_uint2(__pack_half2(Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 20)], Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 20)]), __pack_half2(Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 20)], Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 20)]));
      ((half2*)(&(__271.x)))->x = (((half2*)(&(__323.x)))->x*((half2*)(&(__324.x)))->x);
      ((half2*)(&(__271.x)))->y = (((half2*)(&(__323.x)))->y*((half2*)(&(__324.x)))->y);
      ((half2*)(&(__271.y)))->x = (((half2*)(&(__323.y)))->x*((half2*)(&(__324.y)))->x);
      ((half2*)(&(__271.y)))->y = (((half2*)(&(__323.y)))->y*((half2*)(&(__324.y)))->y);
    *(uint2*)(B_decode_shared + ((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 4)) + 800)) = __271;
    uint2 __325;
      int4 __326;
      int __327;
        int __328;
          int4 __329;
            int4 __330 = make_int4(((((((((int)blockIdx.x) % 160) * 221184) + ((((int)threadIdx.x) >> 3) * 6912)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 165888), ((((((((int)blockIdx.x) % 160) * 221184) + ((((int)threadIdx.x) >> 3) * 6912)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 165888), ((((((((int)blockIdx.x) % 160) * 221184) + ((((int)threadIdx.x) >> 3) * 6912)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 165888), ((((((((int)blockIdx.x) % 160) * 221184) + ((((int)threadIdx.x) >> 3) * 6912)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 165888));
            int4 __331;
              int4 __332 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __333 = make_int4(2, 2, 2, 2);
              __331.x = (__332.x%__333.x);
              __331.y = (__332.y%__333.y);
              __331.z = (__332.z%__333.z);
              __331.w = (__332.w%__333.w);
            int4 __334;
              int4 __335 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __336 = make_int4(2, 2, 2, 2);
              __334.x = (__335.x/__336.x);
              __334.y = (__335.y/__336.y);
              __334.z = (__335.z/__336.z);
              __334.w = (__335.w/__336.w);
            int4 __337;
            ushort4 __338;
              ushort4 __339;
                ushort4 __340;
                  int4 __341 = make_int4(2, 2, 2, 2);
                  int4 __342 = make_int4(0, 0, 0, 0);
                  __340.x = (__341.x>=__342.x);
                  __340.y = (__341.y>=__342.y);
                  __340.z = (__341.z>=__342.z);
                  __340.w = (__341.w>=__342.w);
                ushort4 __343;
                  int4 __344 = make_int4(0, 0, 0, 0);
                  __343.x = (__331.x>=__344.x);
                  __343.y = (__331.y>=__344.y);
                  __343.z = (__331.z>=__344.z);
                  __343.w = (__331.w>=__344.w);
                __339.x = (__340.x&&__343.x);
                __339.y = (__340.y&&__343.y);
                __339.z = (__340.z&&__343.z);
                __339.w = (__340.w&&__343.w);
              ushort4 __345;
                ushort4 __346;
                  int4 __347 = make_int4(2, 2, 2, 2);
                  int4 __348 = make_int4(0, 0, 0, 0);
                  __346.x = (__347.x<__348.x);
                  __346.y = (__347.y<__348.y);
                  __346.z = (__347.z<__348.z);
                  __346.w = (__347.w<__348.w);
                ushort4 __349;
                  int4 __350 = make_int4(0, 0, 0, 0);
                  __349.x = (__331.x<=__350.x);
                  __349.y = (__331.y<=__350.y);
                  __349.z = (__331.z<=__350.z);
                  __349.w = (__331.w<=__350.w);
                __345.x = (__346.x&&__349.x);
                __345.y = (__346.y&&__349.y);
                __345.z = (__346.z&&__349.z);
                __345.w = (__346.w&&__349.w);
              __338.x = (__339.x||__345.x);
              __338.y = (__339.y||__345.y);
              __338.z = (__339.z||__345.z);
              __338.w = (__339.w||__345.w);
            int4 __351;
              int4 __352 = make_int4(1, 1, 1, 1);
              __351.x = (__334.x-__352.x);
              __351.y = (__334.y-__352.y);
              __351.z = (__334.z-__352.z);
              __351.w = (__334.w-__352.w);
            __337.x = (bool(__338.x)?__334.x:__351.x);
            __337.y = (bool(__338.y)?__334.y:__351.y);
            __337.z = (bool(__338.z)?__334.z:__351.z);
            __337.w = (bool(__338.w)?__334.w:__351.w);
            __329.x = (__330.x+__337.x);
            __329.y = (__330.y+__337.y);
            __329.z = (__330.z+__337.z);
            __329.w = (__330.w+__337.w);
          int __353 = ((0x000000ff << 0) & (B[__329.x] << 0))|((0x000000ff << 8) & (B[__329.y] << 8))|((0x000000ff << 16) & (B[__329.z] << 16))|((0x000000ff << 24) & (B[__329.w] << 24));
          int __354;
          int4 __355;
            int4 __356;
              int4 __357 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __358 = make_int4(2, 2, 2, 2);
              __356.x = (__357.x%__358.x);
              __356.y = (__357.y%__358.y);
              __356.z = (__357.z%__358.z);
              __356.w = (__357.w%__358.w);
            int4 __359;
            ushort4 __360;
              ushort4 __361;
                ushort4 __362;
                  int4 __363 = make_int4(2, 2, 2, 2);
                  int4 __364 = make_int4(0, 0, 0, 0);
                  __362.x = (__363.x>=__364.x);
                  __362.y = (__363.y>=__364.y);
                  __362.z = (__363.z>=__364.z);
                  __362.w = (__363.w>=__364.w);
                ushort4 __365;
                  int4 __366 = make_int4(0, 0, 0, 0);
                  __365.x = (__356.x>=__366.x);
                  __365.y = (__356.y>=__366.y);
                  __365.z = (__356.z>=__366.z);
                  __365.w = (__356.w>=__366.w);
                __361.x = (__362.x&&__365.x);
                __361.y = (__362.y&&__365.y);
                __361.z = (__362.z&&__365.z);
                __361.w = (__362.w&&__365.w);
              ushort4 __367;
                ushort4 __368;
                  int4 __369 = make_int4(2, 2, 2, 2);
                  int4 __370 = make_int4(0, 0, 0, 0);
                  __368.x = (__369.x<__370.x);
                  __368.y = (__369.y<__370.y);
                  __368.z = (__369.z<__370.z);
                  __368.w = (__369.w<__370.w);
                ushort4 __371;
                  int4 __372 = make_int4(0, 0, 0, 0);
                  __371.x = (__356.x<=__372.x);
                  __371.y = (__356.y<=__372.y);
                  __371.z = (__356.z<=__372.z);
                  __371.w = (__356.w<=__372.w);
                __367.x = (__368.x&&__371.x);
                __367.y = (__368.y&&__371.y);
                __367.z = (__368.z&&__371.z);
                __367.w = (__368.w&&__371.w);
              __360.x = (__361.x||__367.x);
              __360.y = (__361.y||__367.y);
              __360.z = (__361.z||__367.z);
              __360.w = (__361.w||__367.w);
            int4 __373;
              int4 __374 = make_int4(2, 2, 2, 2);
              __373.x = (__356.x+__374.x);
              __373.y = (__356.y+__374.y);
              __373.z = (__356.z+__374.z);
              __373.w = (__356.w+__374.w);
            __359.x = (bool(__360.x)?__356.x:__373.x);
            __359.y = (bool(__360.y)?__356.y:__373.y);
            __359.z = (bool(__360.z)?__356.z:__373.z);
            __359.w = (bool(__360.w)?__356.w:__373.w);
            int4 __375 = make_int4(4, 4, 4, 4);
            __355.x = (__359.x*__375.x);
            __355.y = (__359.y*__375.y);
            __355.z = (__359.z*__375.z);
            __355.w = (__359.w*__375.w);
          __354=((signed char)(__355.x) << 0);
          __354=__354 & ~(0x000000ff << 8) |((signed char)(__355.y) << 8);
          __354=__354 & ~(0x000000ff << 16) |((signed char)(__355.z) << 16);
          __354=__354 & ~(0x000000ff << 24) |((signed char)(__355.w) << 24);
          __328=((((char)(__353 >> 0)) >> ((char)(__354 >> 0))) << 0);
          __328=__328 & ~(0x000000ff << 8) |((((char)(__353 >> 8)) >> ((char)(__354 >> 8))) << 8);
          __328=__328 & ~(0x000000ff << 16) |((((char)(__353 >> 16)) >> ((char)(__354 >> 16))) << 16);
          __328=__328 & ~(0x000000ff << 24) |((((char)(__353 >> 24)) >> ((char)(__354 >> 24))) << 24);
        int __376 = (int)252645135;
        __327=((((char)(__328 >> 0)) & ((char)(__376 >> 0))) << 0);
        __327=__327 & ~(0x000000ff << 8) |((((char)(__328 >> 8)) & ((char)(__376 >> 8))) << 8);
        __327=__327 & ~(0x000000ff << 16) |((((char)(__328 >> 16)) & ((char)(__376 >> 16))) << 16);
        __327=__327 & ~(0x000000ff << 24) |((((char)(__328 >> 24)) & ((char)(__376 >> 24))) << 24);
      __326.x = (int)(((char)(__327 >> 0)));
      __326.y = (int)(((char)(__327 >> 8)));
      __326.z = (int)(((char)(__327 >> 16)));
      __326.w = (int)(((char)(__327 >> 24)));
      uint2 __377 = make_uint2(__pack_half2(LUT[__326.x],LUT[__326.y]),__pack_half2(LUT[__326.z],LUT[__326.w]));
      uint2 __378 = make_uint2(__pack_half2(Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 24)], Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 24)]), __pack_half2(Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 24)], Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 24)]));
      ((half2*)(&(__325.x)))->x = (((half2*)(&(__377.x)))->x*((half2*)(&(__378.x)))->x);
      ((half2*)(&(__325.x)))->y = (((half2*)(&(__377.x)))->y*((half2*)(&(__378.x)))->y);
      ((half2*)(&(__325.y)))->x = (((half2*)(&(__377.y)))->x*((half2*)(&(__378.y)))->x);
      ((half2*)(&(__325.y)))->y = (((half2*)(&(__377.y)))->y*((half2*)(&(__378.y)))->y);
    *(uint2*)(B_decode_shared + ((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 4)) + 960)) = __325;
    uint2 __379;
      int4 __380;
      int __381;
        int __382;
          int4 __383;
            int4 __384 = make_int4(((((((((int)blockIdx.x) % 160) * 221184) + ((((int)threadIdx.x) >> 3) * 6912)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 193536), ((((((((int)blockIdx.x) % 160) * 221184) + ((((int)threadIdx.x) >> 3) * 6912)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 193536), ((((((((int)blockIdx.x) % 160) * 221184) + ((((int)threadIdx.x) >> 3) * 6912)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 193536), ((((((((int)blockIdx.x) % 160) * 221184) + ((((int)threadIdx.x) >> 3) * 6912)) + (k_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 193536));
            int4 __385;
              int4 __386 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __387 = make_int4(2, 2, 2, 2);
              __385.x = (__386.x%__387.x);
              __385.y = (__386.y%__387.y);
              __385.z = (__386.z%__387.z);
              __385.w = (__386.w%__387.w);
            int4 __388;
              int4 __389 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __390 = make_int4(2, 2, 2, 2);
              __388.x = (__389.x/__390.x);
              __388.y = (__389.y/__390.y);
              __388.z = (__389.z/__390.z);
              __388.w = (__389.w/__390.w);
            int4 __391;
            ushort4 __392;
              ushort4 __393;
                ushort4 __394;
                  int4 __395 = make_int4(2, 2, 2, 2);
                  int4 __396 = make_int4(0, 0, 0, 0);
                  __394.x = (__395.x>=__396.x);
                  __394.y = (__395.y>=__396.y);
                  __394.z = (__395.z>=__396.z);
                  __394.w = (__395.w>=__396.w);
                ushort4 __397;
                  int4 __398 = make_int4(0, 0, 0, 0);
                  __397.x = (__385.x>=__398.x);
                  __397.y = (__385.y>=__398.y);
                  __397.z = (__385.z>=__398.z);
                  __397.w = (__385.w>=__398.w);
                __393.x = (__394.x&&__397.x);
                __393.y = (__394.y&&__397.y);
                __393.z = (__394.z&&__397.z);
                __393.w = (__394.w&&__397.w);
              ushort4 __399;
                ushort4 __400;
                  int4 __401 = make_int4(2, 2, 2, 2);
                  int4 __402 = make_int4(0, 0, 0, 0);
                  __400.x = (__401.x<__402.x);
                  __400.y = (__401.y<__402.y);
                  __400.z = (__401.z<__402.z);
                  __400.w = (__401.w<__402.w);
                ushort4 __403;
                  int4 __404 = make_int4(0, 0, 0, 0);
                  __403.x = (__385.x<=__404.x);
                  __403.y = (__385.y<=__404.y);
                  __403.z = (__385.z<=__404.z);
                  __403.w = (__385.w<=__404.w);
                __399.x = (__400.x&&__403.x);
                __399.y = (__400.y&&__403.y);
                __399.z = (__400.z&&__403.z);
                __399.w = (__400.w&&__403.w);
              __392.x = (__393.x||__399.x);
              __392.y = (__393.y||__399.y);
              __392.z = (__393.z||__399.z);
              __392.w = (__393.w||__399.w);
            int4 __405;
              int4 __406 = make_int4(1, 1, 1, 1);
              __405.x = (__388.x-__406.x);
              __405.y = (__388.y-__406.y);
              __405.z = (__388.z-__406.z);
              __405.w = (__388.w-__406.w);
            __391.x = (bool(__392.x)?__388.x:__405.x);
            __391.y = (bool(__392.y)?__388.y:__405.y);
            __391.z = (bool(__392.z)?__388.z:__405.z);
            __391.w = (bool(__392.w)?__388.w:__405.w);
            __383.x = (__384.x+__391.x);
            __383.y = (__384.y+__391.y);
            __383.z = (__384.z+__391.z);
            __383.w = (__384.w+__391.w);
          int __407 = ((0x000000ff << 0) & (B[__383.x] << 0))|((0x000000ff << 8) & (B[__383.y] << 8))|((0x000000ff << 16) & (B[__383.z] << 16))|((0x000000ff << 24) & (B[__383.w] << 24));
          int __408;
          int4 __409;
            int4 __410;
              int4 __411 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __412 = make_int4(2, 2, 2, 2);
              __410.x = (__411.x%__412.x);
              __410.y = (__411.y%__412.y);
              __410.z = (__411.z%__412.z);
              __410.w = (__411.w%__412.w);
            int4 __413;
            ushort4 __414;
              ushort4 __415;
                ushort4 __416;
                  int4 __417 = make_int4(2, 2, 2, 2);
                  int4 __418 = make_int4(0, 0, 0, 0);
                  __416.x = (__417.x>=__418.x);
                  __416.y = (__417.y>=__418.y);
                  __416.z = (__417.z>=__418.z);
                  __416.w = (__417.w>=__418.w);
                ushort4 __419;
                  int4 __420 = make_int4(0, 0, 0, 0);
                  __419.x = (__410.x>=__420.x);
                  __419.y = (__410.y>=__420.y);
                  __419.z = (__410.z>=__420.z);
                  __419.w = (__410.w>=__420.w);
                __415.x = (__416.x&&__419.x);
                __415.y = (__416.y&&__419.y);
                __415.z = (__416.z&&__419.z);
                __415.w = (__416.w&&__419.w);
              ushort4 __421;
                ushort4 __422;
                  int4 __423 = make_int4(2, 2, 2, 2);
                  int4 __424 = make_int4(0, 0, 0, 0);
                  __422.x = (__423.x<__424.x);
                  __422.y = (__423.y<__424.y);
                  __422.z = (__423.z<__424.z);
                  __422.w = (__423.w<__424.w);
                ushort4 __425;
                  int4 __426 = make_int4(0, 0, 0, 0);
                  __425.x = (__410.x<=__426.x);
                  __425.y = (__410.y<=__426.y);
                  __425.z = (__410.z<=__426.z);
                  __425.w = (__410.w<=__426.w);
                __421.x = (__422.x&&__425.x);
                __421.y = (__422.y&&__425.y);
                __421.z = (__422.z&&__425.z);
                __421.w = (__422.w&&__425.w);
              __414.x = (__415.x||__421.x);
              __414.y = (__415.y||__421.y);
              __414.z = (__415.z||__421.z);
              __414.w = (__415.w||__421.w);
            int4 __427;
              int4 __428 = make_int4(2, 2, 2, 2);
              __427.x = (__410.x+__428.x);
              __427.y = (__410.y+__428.y);
              __427.z = (__410.z+__428.z);
              __427.w = (__410.w+__428.w);
            __413.x = (bool(__414.x)?__410.x:__427.x);
            __413.y = (bool(__414.y)?__410.y:__427.y);
            __413.z = (bool(__414.z)?__410.z:__427.z);
            __413.w = (bool(__414.w)?__410.w:__427.w);
            int4 __429 = make_int4(4, 4, 4, 4);
            __409.x = (__413.x*__429.x);
            __409.y = (__413.y*__429.y);
            __409.z = (__413.z*__429.z);
            __409.w = (__413.w*__429.w);
          __408=((signed char)(__409.x) << 0);
          __408=__408 & ~(0x000000ff << 8) |((signed char)(__409.y) << 8);
          __408=__408 & ~(0x000000ff << 16) |((signed char)(__409.z) << 16);
          __408=__408 & ~(0x000000ff << 24) |((signed char)(__409.w) << 24);
          __382=((((char)(__407 >> 0)) >> ((char)(__408 >> 0))) << 0);
          __382=__382 & ~(0x000000ff << 8) |((((char)(__407 >> 8)) >> ((char)(__408 >> 8))) << 8);
          __382=__382 & ~(0x000000ff << 16) |((((char)(__407 >> 16)) >> ((char)(__408 >> 16))) << 16);
          __382=__382 & ~(0x000000ff << 24) |((((char)(__407 >> 24)) >> ((char)(__408 >> 24))) << 24);
        int __430 = (int)252645135;
        __381=((((char)(__382 >> 0)) & ((char)(__430 >> 0))) << 0);
        __381=__381 & ~(0x000000ff << 8) |((((char)(__382 >> 8)) & ((char)(__430 >> 8))) << 8);
        __381=__381 & ~(0x000000ff << 16) |((((char)(__382 >> 16)) & ((char)(__430 >> 16))) << 16);
        __381=__381 & ~(0x000000ff << 24) |((((char)(__382 >> 24)) & ((char)(__430 >> 24))) << 24);
      __380.x = (int)(((char)(__381 >> 0)));
      __380.y = (int)(((char)(__381 >> 8)));
      __380.z = (int)(((char)(__381 >> 16)));
      __380.w = (int)(((char)(__381 >> 24)));
      uint2 __431 = make_uint2(__pack_half2(LUT[__380.x],LUT[__380.y]),__pack_half2(LUT[__380.z],LUT[__380.w]));
      uint2 __432 = make_uint2(__pack_half2(Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 28)], Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 28)]), __pack_half2(Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 28)], Scales[(((((k_outer >> 2) * 5120) + ((((int)blockIdx.x) % 160) * 32)) + (((int)threadIdx.x) >> 3)) + 28)]));
      ((half2*)(&(__379.x)))->x = (((half2*)(&(__431.x)))->x*((half2*)(&(__432.x)))->x);
      ((half2*)(&(__379.x)))->y = (((half2*)(&(__431.x)))->y*((half2*)(&(__432.x)))->y);
      ((half2*)(&(__379.y)))->x = (((half2*)(&(__431.y)))->x*((half2*)(&(__432.y)))->x);
      ((half2*)(&(__379.y)))->y = (((half2*)(&(__431.y)))->y*((half2*)(&(__432.y)))->y);
    *(uint2*)(B_decode_shared + ((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 4)) + 1120)) = __379;
    __syncthreads();
    for (int k_inner_outer = 0; k_inner_outer < 2; ++k_inner_outer) {
      nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[0], (&(A_shared[(k_inner_outer * 16)])), 40);
      nvcuda::wmma::load_matrix_sync(B_decode_shared_wmma_matrix_b[0], (&(B_decode_shared[(k_inner_outer * 16)])), 40);
      nvcuda::wmma::mma_sync(C_wmma_accumulator[0], A_shared_wmma_matrix_a[0], B_decode_shared_wmma_matrix_b[0], C_wmma_accumulator[0]);
    }
  }
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(C[(((((int)blockIdx.x) / 160) * 40960) + ((((int)blockIdx.x) % 160) * 32))])), C_wmma_accumulator[0], 5120, nvcuda::wmma::mem_row_major);
  __syncthreads();
}



__global__ void __launch_bounds__(128) cutlass_kernel_fp16_nf4_fp16_m16n13824k5120_nt_16x32x32_16x8x16_(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ C) {
            int8_t* B = ((int8_t *)QB);
	 half* LUT = (half *)((int8_t *)QB + 35389440);
	 half* Scales = (half *)((int8_t *)QB + 35389440 + 32);                 
            // const dim3 GridDim(432, 1, 1);
            // const dim3 BlockDim(32, 4, 1);
            // cutlass_kernel_fp16_nf4_fp16_m16n13824k5120_nt_16x32x32_16x8x16_<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
  
  __shared__ half LUT_shared[16];
    __shared__ half A_shared[1024];
  __shared__ half B_decode_shared[2560];
  __shared__ signed char B_shared[1280];
  half B_decode_local[8];
  signed char B_local[2];
  half B_decode_local_1[8];
  signed char B_local_1[2];
  if (((int)threadIdx.x) < 16) {
    LUT_shared[((int)threadIdx.x)] = LUT[((int)threadIdx.x)];
  }
  ALLOCATE_CUTLASS_OBJECT(C_cutlass_warp_mma, cutlass::gemm::warp::GemmTensorOp<
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<16, 32>,
    cutlass::layout::ColumnMajor
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
        :: "r"(addr), "l"((void*)(A + (((((int)threadIdx.y) * 40960) + ((((int)threadIdx.x) >> 2) * 5120)) + ((((int)threadIdx.x) & 3) * 8)))), "n"(16)
    );
  }
    }
  }
  #pragma unroll
  for (int ax0_ax1_0_fused_0_0 = 0; ax0_ax1_0_fused_0_0 < 1; ++ax0_ax1_0_fused_0_0) {
    __syncthreads();

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
        :: "r"(addr), "l"((void*)(B + ((((((int)blockIdx.x) * 81920) + (((int)threadIdx.y) * 20480)) + ((((int)threadIdx.x) >> 2) * 2560)) + ((((int)threadIdx.x) & 3) * 4)))), "n"(4)
    );
  }
    __syncthreads();
    for (int ax0_0 = 0; ax0_0 < 2; ++ax0_0) {
      *(char2*)(B_local + 0) = *(char2*)(B_shared + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 4)) + (ax0_0 * 2)));
      uint2 __1;
        int4 __2;
        int __3;
          int __4;
            int4 __5;
              int4 __6 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __7 = make_int4(2, 2, 2, 2);
              __5.x = (__6.x%__7.x);
              __5.y = (__6.y%__7.y);
              __5.z = (__6.z%__7.z);
              __5.w = (__6.w%__7.w);
            int4 __8;
              int4 __9 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __10 = make_int4(2, 2, 2, 2);
              __8.x = (__9.x/__10.x);
              __8.y = (__9.y/__10.y);
              __8.z = (__9.z/__10.z);
              __8.w = (__9.w/__10.w);
            int4 __11;
            ushort4 __12;
              ushort4 __13;
                ushort4 __14;
                  int4 __15 = make_int4(2, 2, 2, 2);
                  int4 __16 = make_int4(0, 0, 0, 0);
                  __14.x = (__15.x>=__16.x);
                  __14.y = (__15.y>=__16.y);
                  __14.z = (__15.z>=__16.z);
                  __14.w = (__15.w>=__16.w);
                ushort4 __17;
                  int4 __18 = make_int4(0, 0, 0, 0);
                  __17.x = (__5.x>=__18.x);
                  __17.y = (__5.y>=__18.y);
                  __17.z = (__5.z>=__18.z);
                  __17.w = (__5.w>=__18.w);
                __13.x = (__14.x&&__17.x);
                __13.y = (__14.y&&__17.y);
                __13.z = (__14.z&&__17.z);
                __13.w = (__14.w&&__17.w);
              ushort4 __19;
                ushort4 __20;
                  int4 __21 = make_int4(2, 2, 2, 2);
                  int4 __22 = make_int4(0, 0, 0, 0);
                  __20.x = (__21.x<__22.x);
                  __20.y = (__21.y<__22.y);
                  __20.z = (__21.z<__22.z);
                  __20.w = (__21.w<__22.w);
                ushort4 __23;
                  int4 __24 = make_int4(0, 0, 0, 0);
                  __23.x = (__5.x<=__24.x);
                  __23.y = (__5.y<=__24.y);
                  __23.z = (__5.z<=__24.z);
                  __23.w = (__5.w<=__24.w);
                __19.x = (__20.x&&__23.x);
                __19.y = (__20.y&&__23.y);
                __19.z = (__20.z&&__23.z);
                __19.w = (__20.w&&__23.w);
              __12.x = (__13.x||__19.x);
              __12.y = (__13.y||__19.y);
              __12.z = (__13.z||__19.z);
              __12.w = (__13.w||__19.w);
            int4 __25;
              int4 __26 = make_int4(1, 1, 1, 1);
              __25.x = (__8.x-__26.x);
              __25.y = (__8.y-__26.y);
              __25.z = (__8.z-__26.z);
              __25.w = (__8.w-__26.w);
            __11.x = (bool(__12.x)?__8.x:__25.x);
            __11.y = (bool(__12.y)?__8.y:__25.y);
            __11.z = (bool(__12.z)?__8.z:__25.z);
            __11.w = (bool(__12.w)?__8.w:__25.w);
            int __27 = ((0x000000ff << 0) & (B_local[__11.x] << 0))|((0x000000ff << 8) & (B_local[__11.y] << 8))|((0x000000ff << 16) & (B_local[__11.z] << 16))|((0x000000ff << 24) & (B_local[__11.w] << 24));
            int __28;
            int4 __29;
              int4 __30;
                int4 __31 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
                int4 __32 = make_int4(2, 2, 2, 2);
                __30.x = (__31.x%__32.x);
                __30.y = (__31.y%__32.y);
                __30.z = (__31.z%__32.z);
                __30.w = (__31.w%__32.w);
              int4 __33;
              ushort4 __34;
                ushort4 __35;
                  ushort4 __36;
                    int4 __37 = make_int4(2, 2, 2, 2);
                    int4 __38 = make_int4(0, 0, 0, 0);
                    __36.x = (__37.x>=__38.x);
                    __36.y = (__37.y>=__38.y);
                    __36.z = (__37.z>=__38.z);
                    __36.w = (__37.w>=__38.w);
                  ushort4 __39;
                    int4 __40 = make_int4(0, 0, 0, 0);
                    __39.x = (__30.x>=__40.x);
                    __39.y = (__30.y>=__40.y);
                    __39.z = (__30.z>=__40.z);
                    __39.w = (__30.w>=__40.w);
                  __35.x = (__36.x&&__39.x);
                  __35.y = (__36.y&&__39.y);
                  __35.z = (__36.z&&__39.z);
                  __35.w = (__36.w&&__39.w);
                ushort4 __41;
                  ushort4 __42;
                    int4 __43 = make_int4(2, 2, 2, 2);
                    int4 __44 = make_int4(0, 0, 0, 0);
                    __42.x = (__43.x<__44.x);
                    __42.y = (__43.y<__44.y);
                    __42.z = (__43.z<__44.z);
                    __42.w = (__43.w<__44.w);
                  ushort4 __45;
                    int4 __46 = make_int4(0, 0, 0, 0);
                    __45.x = (__30.x<=__46.x);
                    __45.y = (__30.y<=__46.y);
                    __45.z = (__30.z<=__46.z);
                    __45.w = (__30.w<=__46.w);
                  __41.x = (__42.x&&__45.x);
                  __41.y = (__42.y&&__45.y);
                  __41.z = (__42.z&&__45.z);
                  __41.w = (__42.w&&__45.w);
                __34.x = (__35.x||__41.x);
                __34.y = (__35.y||__41.y);
                __34.z = (__35.z||__41.z);
                __34.w = (__35.w||__41.w);
              int4 __47;
                int4 __48 = make_int4(2, 2, 2, 2);
                __47.x = (__30.x+__48.x);
                __47.y = (__30.y+__48.y);
                __47.z = (__30.z+__48.z);
                __47.w = (__30.w+__48.w);
              __33.x = (bool(__34.x)?__30.x:__47.x);
              __33.y = (bool(__34.y)?__30.y:__47.y);
              __33.z = (bool(__34.z)?__30.z:__47.z);
              __33.w = (bool(__34.w)?__30.w:__47.w);
              int4 __49 = make_int4(4, 4, 4, 4);
              __29.x = (__33.x*__49.x);
              __29.y = (__33.y*__49.y);
              __29.z = (__33.z*__49.z);
              __29.w = (__33.w*__49.w);
            __28=((signed char)(__29.x) << 0);
            __28=__28 & ~(0x000000ff << 8) |((signed char)(__29.y) << 8);
            __28=__28 & ~(0x000000ff << 16) |((signed char)(__29.z) << 16);
            __28=__28 & ~(0x000000ff << 24) |((signed char)(__29.w) << 24);
            __4=((((char)(__27 >> 0)) >> ((char)(__28 >> 0))) << 0);
            __4=__4 & ~(0x000000ff << 8) |((((char)(__27 >> 8)) >> ((char)(__28 >> 8))) << 8);
            __4=__4 & ~(0x000000ff << 16) |((((char)(__27 >> 16)) >> ((char)(__28 >> 16))) << 16);
            __4=__4 & ~(0x000000ff << 24) |((((char)(__27 >> 24)) >> ((char)(__28 >> 24))) << 24);
          int __50 = (int)252645135;
          __3=((((char)(__4 >> 0)) & ((char)(__50 >> 0))) << 0);
          __3=__3 & ~(0x000000ff << 8) |((((char)(__4 >> 8)) & ((char)(__50 >> 8))) << 8);
          __3=__3 & ~(0x000000ff << 16) |((((char)(__4 >> 16)) & ((char)(__50 >> 16))) << 16);
          __3=__3 & ~(0x000000ff << 24) |((((char)(__4 >> 24)) & ((char)(__50 >> 24))) << 24);
        __2.x = (int)(((char)(__3 >> 0)));
        __2.y = (int)(((char)(__3 >> 8)));
        __2.z = (int)(((char)(__3 >> 16)));
        __2.w = (int)(((char)(__3 >> 24)));
        uint2 __51 = make_uint2(__pack_half2(LUT_shared[__2.x],LUT_shared[__2.y]),__pack_half2(LUT_shared[__2.z],LUT_shared[__2.w]));
        uint2 __52 = make_uint2(__pack_half2(Scales[(((((int)blockIdx.x) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))], Scales[(((((int)blockIdx.x) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))]), __pack_half2(Scales[(((((int)blockIdx.x) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))], Scales[(((((int)blockIdx.x) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))]));
        ((half2*)(&(__1.x)))->x = (((half2*)(&(__51.x)))->x*((half2*)(&(__52.x)))->x);
        ((half2*)(&(__1.x)))->y = (((half2*)(&(__51.x)))->y*((half2*)(&(__52.x)))->y);
        ((half2*)(&(__1.y)))->x = (((half2*)(&(__51.y)))->x*((half2*)(&(__52.y)))->x);
        ((half2*)(&(__1.y)))->y = (((half2*)(&(__51.y)))->y*((half2*)(&(__52.y)))->y);
      *(uint2*)(B_decode_local + (ax0_0 * 4)) = __1;
    }
    *(uint4*)(B_decode_shared + (((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) = *(uint4*)(B_decode_local + 0);
  }
__asm__ __volatile__("cp.async.commit_group;");

  #pragma unroll
  for (int ax0_ax1_fused_0_0_0_1 = 0; ax0_ax1_fused_0_0_0_1 < 1; ++ax0_ax1_fused_0_0_0_1) {
    if (((int)threadIdx.y) < 2) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + (((((((int)threadIdx.y) * 256) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 512))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(A_shared + (((((((int)threadIdx.y) * 256) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 512)))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((int)threadIdx.y) * 40960) + ((((int)threadIdx.x) >> 2) * 5120)) + ((((int)threadIdx.x) & 3) * 8)) + 32))), "n"(16)
    );
  }
    }
  }
  #pragma unroll
  for (int ax0_ax1_0_fused_0_0_1 = 0; ax0_ax1_0_fused_0_0_1 < 1; ++ax0_ax1_0_fused_0_0_1) {
    __syncthreads();

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
        :: "r"(addr), "l"((void*)(B + (((((((int)blockIdx.x) * 81920) + (((int)threadIdx.y) * 20480)) + ((((int)threadIdx.x) >> 2) * 2560)) + ((((int)threadIdx.x) & 3) * 4)) + 16))), "n"(4)
    );
  }
    __syncthreads();
    for (int ax0_0_1 = 0; ax0_0_1 < 2; ++ax0_0_1) {
      *(char2*)(B_local + 0) = *(char2*)(B_shared + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 4)) + (ax0_0_1 * 2)));
      uint2 __53;
        int4 __54;
        int __55;
          int __56;
            int4 __57;
              int4 __58 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __59 = make_int4(2, 2, 2, 2);
              __57.x = (__58.x%__59.x);
              __57.y = (__58.y%__59.y);
              __57.z = (__58.z%__59.z);
              __57.w = (__58.w%__59.w);
            int4 __60;
              int4 __61 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
              int4 __62 = make_int4(2, 2, 2, 2);
              __60.x = (__61.x/__62.x);
              __60.y = (__61.y/__62.y);
              __60.z = (__61.z/__62.z);
              __60.w = (__61.w/__62.w);
            int4 __63;
            ushort4 __64;
              ushort4 __65;
                ushort4 __66;
                  int4 __67 = make_int4(2, 2, 2, 2);
                  int4 __68 = make_int4(0, 0, 0, 0);
                  __66.x = (__67.x>=__68.x);
                  __66.y = (__67.y>=__68.y);
                  __66.z = (__67.z>=__68.z);
                  __66.w = (__67.w>=__68.w);
                ushort4 __69;
                  int4 __70 = make_int4(0, 0, 0, 0);
                  __69.x = (__57.x>=__70.x);
                  __69.y = (__57.y>=__70.y);
                  __69.z = (__57.z>=__70.z);
                  __69.w = (__57.w>=__70.w);
                __65.x = (__66.x&&__69.x);
                __65.y = (__66.y&&__69.y);
                __65.z = (__66.z&&__69.z);
                __65.w = (__66.w&&__69.w);
              ushort4 __71;
                ushort4 __72;
                  int4 __73 = make_int4(2, 2, 2, 2);
                  int4 __74 = make_int4(0, 0, 0, 0);
                  __72.x = (__73.x<__74.x);
                  __72.y = (__73.y<__74.y);
                  __72.z = (__73.z<__74.z);
                  __72.w = (__73.w<__74.w);
                ushort4 __75;
                  int4 __76 = make_int4(0, 0, 0, 0);
                  __75.x = (__57.x<=__76.x);
                  __75.y = (__57.y<=__76.y);
                  __75.z = (__57.z<=__76.z);
                  __75.w = (__57.w<=__76.w);
                __71.x = (__72.x&&__75.x);
                __71.y = (__72.y&&__75.y);
                __71.z = (__72.z&&__75.z);
                __71.w = (__72.w&&__75.w);
              __64.x = (__65.x||__71.x);
              __64.y = (__65.y||__71.y);
              __64.z = (__65.z||__71.z);
              __64.w = (__65.w||__71.w);
            int4 __77;
              int4 __78 = make_int4(1, 1, 1, 1);
              __77.x = (__60.x-__78.x);
              __77.y = (__60.y-__78.y);
              __77.z = (__60.z-__78.z);
              __77.w = (__60.w-__78.w);
            __63.x = (bool(__64.x)?__60.x:__77.x);
            __63.y = (bool(__64.y)?__60.y:__77.y);
            __63.z = (bool(__64.z)?__60.z:__77.z);
            __63.w = (bool(__64.w)?__60.w:__77.w);
            int __79 = ((0x000000ff << 0) & (B_local[__63.x] << 0))|((0x000000ff << 8) & (B_local[__63.y] << 8))|((0x000000ff << 16) & (B_local[__63.z] << 16))|((0x000000ff << 24) & (B_local[__63.w] << 24));
            int __80;
            int4 __81;
              int4 __82;
                int4 __83 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
                int4 __84 = make_int4(2, 2, 2, 2);
                __82.x = (__83.x%__84.x);
                __82.y = (__83.y%__84.y);
                __82.z = (__83.z%__84.z);
                __82.w = (__83.w%__84.w);
              int4 __85;
              ushort4 __86;
                ushort4 __87;
                  ushort4 __88;
                    int4 __89 = make_int4(2, 2, 2, 2);
                    int4 __90 = make_int4(0, 0, 0, 0);
                    __88.x = (__89.x>=__90.x);
                    __88.y = (__89.y>=__90.y);
                    __88.z = (__89.z>=__90.z);
                    __88.w = (__89.w>=__90.w);
                  ushort4 __91;
                    int4 __92 = make_int4(0, 0, 0, 0);
                    __91.x = (__82.x>=__92.x);
                    __91.y = (__82.y>=__92.y);
                    __91.z = (__82.z>=__92.z);
                    __91.w = (__82.w>=__92.w);
                  __87.x = (__88.x&&__91.x);
                  __87.y = (__88.y&&__91.y);
                  __87.z = (__88.z&&__91.z);
                  __87.w = (__88.w&&__91.w);
                ushort4 __93;
                  ushort4 __94;
                    int4 __95 = make_int4(2, 2, 2, 2);
                    int4 __96 = make_int4(0, 0, 0, 0);
                    __94.x = (__95.x<__96.x);
                    __94.y = (__95.y<__96.y);
                    __94.z = (__95.z<__96.z);
                    __94.w = (__95.w<__96.w);
                  ushort4 __97;
                    int4 __98 = make_int4(0, 0, 0, 0);
                    __97.x = (__82.x<=__98.x);
                    __97.y = (__82.y<=__98.y);
                    __97.z = (__82.z<=__98.z);
                    __97.w = (__82.w<=__98.w);
                  __93.x = (__94.x&&__97.x);
                  __93.y = (__94.y&&__97.y);
                  __93.z = (__94.z&&__97.z);
                  __93.w = (__94.w&&__97.w);
                __86.x = (__87.x||__93.x);
                __86.y = (__87.y||__93.y);
                __86.z = (__87.z||__93.z);
                __86.w = (__87.w||__93.w);
              int4 __99;
                int4 __100 = make_int4(2, 2, 2, 2);
                __99.x = (__82.x+__100.x);
                __99.y = (__82.y+__100.y);
                __99.z = (__82.z+__100.z);
                __99.w = (__82.w+__100.w);
              __85.x = (bool(__86.x)?__82.x:__99.x);
              __85.y = (bool(__86.y)?__82.y:__99.y);
              __85.z = (bool(__86.z)?__82.z:__99.z);
              __85.w = (bool(__86.w)?__82.w:__99.w);
              int4 __101 = make_int4(4, 4, 4, 4);
              __81.x = (__85.x*__101.x);
              __81.y = (__85.y*__101.y);
              __81.z = (__85.z*__101.z);
              __81.w = (__85.w*__101.w);
            __80=((signed char)(__81.x) << 0);
            __80=__80 & ~(0x000000ff << 8) |((signed char)(__81.y) << 8);
            __80=__80 & ~(0x000000ff << 16) |((signed char)(__81.z) << 16);
            __80=__80 & ~(0x000000ff << 24) |((signed char)(__81.w) << 24);
            __56=((((char)(__79 >> 0)) >> ((char)(__80 >> 0))) << 0);
            __56=__56 & ~(0x000000ff << 8) |((((char)(__79 >> 8)) >> ((char)(__80 >> 8))) << 8);
            __56=__56 & ~(0x000000ff << 16) |((((char)(__79 >> 16)) >> ((char)(__80 >> 16))) << 16);
            __56=__56 & ~(0x000000ff << 24) |((((char)(__79 >> 24)) >> ((char)(__80 >> 24))) << 24);
          int __102 = (int)252645135;
          __55=((((char)(__56 >> 0)) & ((char)(__102 >> 0))) << 0);
          __55=__55 & ~(0x000000ff << 8) |((((char)(__56 >> 8)) & ((char)(__102 >> 8))) << 8);
          __55=__55 & ~(0x000000ff << 16) |((((char)(__56 >> 16)) & ((char)(__102 >> 16))) << 16);
          __55=__55 & ~(0x000000ff << 24) |((((char)(__56 >> 24)) & ((char)(__102 >> 24))) << 24);
        __54.x = (int)(((char)(__55 >> 0)));
        __54.y = (int)(((char)(__55 >> 8)));
        __54.z = (int)(((char)(__55 >> 16)));
        __54.w = (int)(((char)(__55 >> 24)));
        uint2 __103 = make_uint2(__pack_half2(LUT_shared[__54.x],LUT_shared[__54.y]),__pack_half2(LUT_shared[__54.z],LUT_shared[__54.w]));
        uint2 __104 = make_uint2(__pack_half2(Scales[(((((int)blockIdx.x) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))], Scales[(((((int)blockIdx.x) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))]), __pack_half2(Scales[(((((int)blockIdx.x) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))], Scales[(((((int)blockIdx.x) * 32) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2))]));
        ((half2*)(&(__53.x)))->x = (((half2*)(&(__103.x)))->x*((half2*)(&(__104.x)))->x);
        ((half2*)(&(__53.x)))->y = (((half2*)(&(__103.x)))->y*((half2*)(&(__104.x)))->y);
        ((half2*)(&(__53.y)))->x = (((half2*)(&(__103.y)))->x*((half2*)(&(__104.y)))->x);
        ((half2*)(&(__53.y)))->y = (((half2*)(&(__103.y)))->y*((half2*)(&(__104.y)))->y);
      *(uint2*)(B_decode_local + (ax0_0_1 * 4)) = __53;
    }
    *(uint4*)(B_decode_shared + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 1280)) = *(uint4*)(B_decode_local + 0);
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 1;");

  __syncthreads();
  call_cutlass_mma_prologue(C_cutlass_warp_mma, (&(A_shared[0])), (&(B_decode_shared[0])), 32, 40);
  call_cutlass_mma_body(C_cutlass_warp_mma);
  for (int k_0 = 0; k_0 < 158; ++k_0) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_0_0_0_2 = 0; ax0_ax1_fused_0_0_0_2 < 1; ++ax0_ax1_fused_0_0_0_2) {
      if (((int)threadIdx.y) < 2) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + ((((((k_0 & 1) * 512) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(A_shared + ((((((k_0 & 1) * 512) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + (((((((int)threadIdx.y) * 40960) + ((((int)threadIdx.x) >> 2) * 5120)) + (k_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 64))), "n"(16)
    );
  }
      }
    }
    #pragma unroll
    for (int ax0_ax1_0_fused_0_0_2 = 0; ax0_ax1_0_fused_0_0_2 < 1; ++ax0_ax1_0_fused_0_0_2) {
      __syncthreads();

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
        :: "r"(addr), "l"((void*)(B + ((((((((int)blockIdx.x) * 81920) + (((int)threadIdx.y) * 20480)) + ((((int)threadIdx.x) >> 2) * 2560)) + (k_0 * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 32))), "n"(4)
    );
  }
      __syncthreads();
      for (int ax0_0_2 = 0; ax0_0_2 < 2; ++ax0_0_2) {
        *(char2*)(B_local_1 + 0) = *(char2*)(B_shared + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 4)) + (ax0_0_2 * 2)));
        uint2 __105;
          int4 __106;
          int __107;
            int __108;
              int4 __109;
                int4 __110 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
                int4 __111 = make_int4(2, 2, 2, 2);
                __109.x = (__110.x%__111.x);
                __109.y = (__110.y%__111.y);
                __109.z = (__110.z%__111.z);
                __109.w = (__110.w%__111.w);
              int4 __112;
                int4 __113 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
                int4 __114 = make_int4(2, 2, 2, 2);
                __112.x = (__113.x/__114.x);
                __112.y = (__113.y/__114.y);
                __112.z = (__113.z/__114.z);
                __112.w = (__113.w/__114.w);
              int4 __115;
              ushort4 __116;
                ushort4 __117;
                  ushort4 __118;
                    int4 __119 = make_int4(2, 2, 2, 2);
                    int4 __120 = make_int4(0, 0, 0, 0);
                    __118.x = (__119.x>=__120.x);
                    __118.y = (__119.y>=__120.y);
                    __118.z = (__119.z>=__120.z);
                    __118.w = (__119.w>=__120.w);
                  ushort4 __121;
                    int4 __122 = make_int4(0, 0, 0, 0);
                    __121.x = (__109.x>=__122.x);
                    __121.y = (__109.y>=__122.y);
                    __121.z = (__109.z>=__122.z);
                    __121.w = (__109.w>=__122.w);
                  __117.x = (__118.x&&__121.x);
                  __117.y = (__118.y&&__121.y);
                  __117.z = (__118.z&&__121.z);
                  __117.w = (__118.w&&__121.w);
                ushort4 __123;
                  ushort4 __124;
                    int4 __125 = make_int4(2, 2, 2, 2);
                    int4 __126 = make_int4(0, 0, 0, 0);
                    __124.x = (__125.x<__126.x);
                    __124.y = (__125.y<__126.y);
                    __124.z = (__125.z<__126.z);
                    __124.w = (__125.w<__126.w);
                  ushort4 __127;
                    int4 __128 = make_int4(0, 0, 0, 0);
                    __127.x = (__109.x<=__128.x);
                    __127.y = (__109.y<=__128.y);
                    __127.z = (__109.z<=__128.z);
                    __127.w = (__109.w<=__128.w);
                  __123.x = (__124.x&&__127.x);
                  __123.y = (__124.y&&__127.y);
                  __123.z = (__124.z&&__127.z);
                  __123.w = (__124.w&&__127.w);
                __116.x = (__117.x||__123.x);
                __116.y = (__117.y||__123.y);
                __116.z = (__117.z||__123.z);
                __116.w = (__117.w||__123.w);
              int4 __129;
                int4 __130 = make_int4(1, 1, 1, 1);
                __129.x = (__112.x-__130.x);
                __129.y = (__112.y-__130.y);
                __129.z = (__112.z-__130.z);
                __129.w = (__112.w-__130.w);
              __115.x = (bool(__116.x)?__112.x:__129.x);
              __115.y = (bool(__116.y)?__112.y:__129.y);
              __115.z = (bool(__116.z)?__112.z:__129.z);
              __115.w = (bool(__116.w)?__112.w:__129.w);
              int __131 = ((0x000000ff << 0) & (B_local_1[__115.x] << 0))|((0x000000ff << 8) & (B_local_1[__115.y] << 8))|((0x000000ff << 16) & (B_local_1[__115.z] << 16))|((0x000000ff << 24) & (B_local_1[__115.w] << 24));
              int __132;
              int4 __133;
                int4 __134;
                  int4 __135 = make_int4((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3));
                  int4 __136 = make_int4(2, 2, 2, 2);
                  __134.x = (__135.x%__136.x);
                  __134.y = (__135.y%__136.y);
                  __134.z = (__135.z%__136.z);
                  __134.w = (__135.w%__136.w);
                int4 __137;
                ushort4 __138;
                  ushort4 __139;
                    ushort4 __140;
                      int4 __141 = make_int4(2, 2, 2, 2);
                      int4 __142 = make_int4(0, 0, 0, 0);
                      __140.x = (__141.x>=__142.x);
                      __140.y = (__141.y>=__142.y);
                      __140.z = (__141.z>=__142.z);
                      __140.w = (__141.w>=__142.w);
                    ushort4 __143;
                      int4 __144 = make_int4(0, 0, 0, 0);
                      __143.x = (__134.x>=__144.x);
                      __143.y = (__134.y>=__144.y);
                      __143.z = (__134.z>=__144.z);
                      __143.w = (__134.w>=__144.w);
                    __139.x = (__140.x&&__143.x);
                    __139.y = (__140.y&&__143.y);
                    __139.z = (__140.z&&__143.z);
                    __139.w = (__140.w&&__143.w);
                  ushort4 __145;
                    ushort4 __146;
                      int4 __147 = make_int4(2, 2, 2, 2);
                      int4 __148 = make_int4(0, 0, 0, 0);
                      __146.x = (__147.x<__148.x);
                      __146.y = (__147.y<__148.y);
                      __146.z = (__147.z<__148.z);
                      __146.w = (__147.w<__148.w);
                    ushort4 __149;
                      int4 __150 = make_int4(0, 0, 0, 0);
                      __149.x = (__134.x<=__150.x);
                      __149.y = (__134.y<=__150.y);
                      __149.z = (__134.z<=__150.z);
                      __149.w = (__134.w<=__150.w);
                    __145.x = (__146.x&&__149.x);
                    __145.y = (__146.y&&__149.y);
                    __145.z = (__146.z&&__149.z);
                    __145.w = (__146.w&&__149.w);
                  __138.x = (__139.x||__145.x);
                  __138.y = (__139.y||__145.y);
                  __138.z = (__139.z||__145.z);
                  __138.w = (__139.w||__145.w);
                int4 __151;
                  int4 __152 = make_int4(2, 2, 2, 2);
                  __151.x = (__134.x+__152.x);
                  __151.y = (__134.y+__152.y);
                  __151.z = (__134.z+__152.z);
                  __151.w = (__134.w+__152.w);
                __137.x = (bool(__138.x)?__134.x:__151.x);
                __137.y = (bool(__138.y)?__134.y:__151.y);
                __137.z = (bool(__138.z)?__134.z:__151.z);
                __137.w = (bool(__138.w)?__134.w:__151.w);
                int4 __153 = make_int4(4, 4, 4, 4);
                __133.x = (__137.x*__153.x);
                __133.y = (__137.y*__153.y);
                __133.z = (__137.z*__153.z);
                __133.w = (__137.w*__153.w);
              __132=((signed char)(__133.x) << 0);
              __132=__132 & ~(0x000000ff << 8) |((signed char)(__133.y) << 8);
              __132=__132 & ~(0x000000ff << 16) |((signed char)(__133.z) << 16);
              __132=__132 & ~(0x000000ff << 24) |((signed char)(__133.w) << 24);
              __108=((((char)(__131 >> 0)) >> ((char)(__132 >> 0))) << 0);
              __108=__108 & ~(0x000000ff << 8) |((((char)(__131 >> 8)) >> ((char)(__132 >> 8))) << 8);
              __108=__108 & ~(0x000000ff << 16) |((((char)(__131 >> 16)) >> ((char)(__132 >> 16))) << 16);
              __108=__108 & ~(0x000000ff << 24) |((((char)(__131 >> 24)) >> ((char)(__132 >> 24))) << 24);
            int __154 = (int)252645135;
            __107=((((char)(__108 >> 0)) & ((char)(__154 >> 0))) << 0);
            __107=__107 & ~(0x000000ff << 8) |((((char)(__108 >> 8)) & ((char)(__154 >> 8))) << 8);
            __107=__107 & ~(0x000000ff << 16) |((((char)(__108 >> 16)) & ((char)(__154 >> 16))) << 16);
            __107=__107 & ~(0x000000ff << 24) |((((char)(__108 >> 24)) & ((char)(__154 >> 24))) << 24);
          __106.x = (int)(((char)(__107 >> 0)));
          __106.y = (int)(((char)(__107 >> 8)));
          __106.z = (int)(((char)(__107 >> 16)));
          __106.w = (int)(((char)(__107 >> 24)));
          uint2 __155 = make_uint2(__pack_half2(LUT_shared[__106.x],LUT_shared[__106.y]),__pack_half2(LUT_shared[__106.z],LUT_shared[__106.w]));
          int4 __156 = make_int4((((((((((k_0 * 32) + ((((int)threadIdx.x) & 3) * 8)) + (ax0_0_2 * 4)) + 64) >> 7) * 13824) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)), (((((((((k_0 * 32) + ((((int)threadIdx.x) & 3) * 8)) + (ax0_0_2 * 4)) + 64) >> 7) * 13824) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)), (((((((((k_0 * 32) + ((((int)threadIdx.x) & 3) * 8)) + (ax0_0_2 * 4)) + 64) >> 7) * 13824) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)), (((((((((k_0 * 32) + ((((int)threadIdx.x) & 3) * 8)) + (ax0_0_2 * 4)) + 64) >> 7) * 13824) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)));
          uint2 __157 = make_uint2(__pack_half2(Scales[__156.x],Scales[__156.y]),__pack_half2(Scales[__156.z],Scales[__156.w]));
          ((half2*)(&(__105.x)))->x = (((half2*)(&(__155.x)))->x*((half2*)(&(__157.x)))->x);
          ((half2*)(&(__105.x)))->y = (((half2*)(&(__155.x)))->y*((half2*)(&(__157.x)))->y);
          ((half2*)(&(__105.y)))->x = (((half2*)(&(__155.y)))->x*((half2*)(&(__157.y)))->x);
          ((half2*)(&(__105.y)))->y = (((half2*)(&(__155.y)))->y*((half2*)(&(__157.y)))->y);
        *(uint2*)(B_decode_local_1 + (ax0_0_2 * 4)) = __105;
      }
      *(uint4*)(B_decode_shared + (((((k_0 & 1) * 1280) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) = *(uint4*)(B_decode_local_1 + 0);
    }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 1;");

    __syncthreads();
    call_cutlass_mma_prologue(C_cutlass_warp_mma, (&(A_shared[(((k_0 + 1) & 1) * 512)])), (&(B_decode_shared[(((k_0 + 1) & 1) * 1280)])), 32, 40);
    call_cutlass_mma_epilogue(C_cutlass_warp_mma);
    call_cutlass_mma_body(C_cutlass_warp_mma);
  }
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  call_cutlass_mma_prologue(C_cutlass_warp_mma, (&(A_shared[512])), (&(B_decode_shared[1280])), 32, 40);
  call_cutlass_mma_epilogue(C_cutlass_warp_mma);
  call_cutlass_mma_body(C_cutlass_warp_mma);
  call_cutlass_mma_epilogue(C_cutlass_warp_mma);
  #pragma unroll
  for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0) {
    *(uint1*)(C + (((((ax1_0 * 110592) + ((((int)threadIdx.x) >> 2) * 13824)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.y) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(C_cutlass_warp_mma + (ax1_0 * 2));
  }
}



 __global__ void __launch_bounds__(128) cutlass_kernel_fp16_nf4_fp16_m1n15360k5120_nt_4x8(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ C) {
            int8_t* B = ((int8_t *)QB);
	 half* LUT = (half *)((int8_t *)QB + 39321600);
	 half* Scales = (half *)((int8_t *)QB + 39321600 + 32);                 
            // const dim3 GridDim(3840, 1, 1);
            // const dim3 BlockDim(32, 4, 1);
            // cutlass_kernel_fp16_nf4_fp16_m1n15360k5120_nt_4x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
  half in_thread_C_local[1];
  half A_local[8];
  signed char B_local[4];
  half red_buf0[1];
  in_thread_C_local[0] = __float2half_rn(0.000000e+00f);
  for (int k_0 = 0; k_0 < 20; ++k_0) {
    *(uint4*)(A_local + 0) = *(uint4*)(A + ((k_0 * 256) + (((int)threadIdx.x) * 8)));
    *(int*)(B_local + 0) = *(int*)(B + ((((((int)blockIdx.x) * 10240) + (((int)threadIdx.y) * 2560)) + (k_0 * 128)) + (((int)threadIdx.x) * 4)));
    for (int k_2 = 0; k_2 < 8; ++k_2) {
      in_thread_C_local[0] = (in_thread_C_local[0] + (A_local[k_2] * (LUT[((int)((B_local[(k_2 >> 1)] >> ((signed char)((k_2 & 1) * 4))) & (signed char)15))] * Scales[((((k_0 * 30720) + ((((int)threadIdx.x) >> 4) * 15360)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.y))])));
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



 __global__ void __launch_bounds__(128) cutlass_kernel_fp16_nf4_fp16_m1n5120k5120_nt_4x8(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ C) {
            int8_t* B = ((int8_t *)QB);
	 half* LUT = (half *)((int8_t *)QB + 13107200);
	 half* Scales = (half *)((int8_t *)QB + 13107200 + 32);                 
            // const dim3 GridDim(1280, 1, 1);
            // const dim3 BlockDim(32, 4, 1);
            // cutlass_kernel_fp16_nf4_fp16_m1n5120k5120_nt_4x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
  half in_thread_C_local[1];
  half A_local[8];
  signed char B_local[4];
  half red_buf0[1];
  in_thread_C_local[0] = __float2half_rn(0.000000e+00f);
  for (int k_0 = 0; k_0 < 20; ++k_0) {
    *(uint4*)(A_local + 0) = *(uint4*)(A + ((k_0 * 256) + (((int)threadIdx.x) * 8)));
    *(int*)(B_local + 0) = *(int*)(B + ((((((int)blockIdx.x) * 10240) + (((int)threadIdx.y) * 2560)) + (k_0 * 128)) + (((int)threadIdx.x) * 4)));
    for (int k_2 = 0; k_2 < 8; ++k_2) {
      in_thread_C_local[0] = (in_thread_C_local[0] + (A_local[k_2] * (LUT[((int)((B_local[(k_2 >> 1)] >> ((signed char)((k_2 & 1) * 4))) & (signed char)15))] * Scales[((((k_0 * 10240) + ((((int)threadIdx.x) >> 4) * 5120)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.y))])));
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



 __global__ void __launch_bounds__(128) cutlass_kernel_fp16_nf4_fp16_m1n5120k13824_nt_4x8(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ C) {
            int8_t* B = ((int8_t *)QB);
	 half* LUT = (half *)((int8_t *)QB + 35389440);
	 half* Scales = (half *)((int8_t *)QB + 35389440 + 32);                 
            // const dim3 GridDim(1280, 1, 1);
            // const dim3 BlockDim(32, 4, 1);
            // cutlass_kernel_fp16_nf4_fp16_m1n5120k13824_nt_4x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
  half in_thread_C_local[1];
  half A_local[8];
  signed char B_local[4];
  half red_buf0[1];
  in_thread_C_local[0] = __float2half_rn(0.000000e+00f);
  for (int k_0 = 0; k_0 < 54; ++k_0) {
    *(uint4*)(A_local + 0) = *(uint4*)(A + ((k_0 * 256) + (((int)threadIdx.x) * 8)));
    *(int*)(B_local + 0) = *(int*)(B + ((((((int)blockIdx.x) * 27648) + (((int)threadIdx.y) * 6912)) + (k_0 * 128)) + (((int)threadIdx.x) * 4)));
    for (int k_2 = 0; k_2 < 8; ++k_2) {
      in_thread_C_local[0] = (in_thread_C_local[0] + (A_local[k_2] * (LUT[((int)((B_local[(k_2 >> 1)] >> ((signed char)((k_2 & 1) * 4))) & (signed char)15))] * Scales[((((k_0 * 10240) + ((((int)threadIdx.x) >> 4) * 5120)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.y))])));
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



 __global__ void __launch_bounds__(128) cutlass_kernel_fp16_nf4_fp16_m1n13824k5120_nt_4x8(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ C) {
            int8_t* B = ((int8_t *)QB);
	 half* LUT = (half *)((int8_t *)QB + 35389440);
	 half* Scales = (half *)((int8_t *)QB + 35389440 + 32);                 
            // const dim3 GridDim(3456, 1, 1);
            // const dim3 BlockDim(32, 4, 1);
            // cutlass_kernel_fp16_nf4_fp16_m1n13824k5120_nt_4x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
  half in_thread_C_local[1];
  half A_local[8];
  signed char B_local[4];
  half red_buf0[1];
  in_thread_C_local[0] = __float2half_rn(0.000000e+00f);
  for (int k_0 = 0; k_0 < 20; ++k_0) {
    *(uint4*)(A_local + 0) = *(uint4*)(A + ((k_0 * 256) + (((int)threadIdx.x) * 8)));
    *(int*)(B_local + 0) = *(int*)(B + ((((((int)blockIdx.x) * 10240) + (((int)threadIdx.y) * 2560)) + (k_0 * 128)) + (((int)threadIdx.x) * 4)));
    for (int k_2 = 0; k_2 < 8; ++k_2) {
      in_thread_C_local[0] = (in_thread_C_local[0] + (A_local[k_2] * (LUT[((int)((B_local[(k_2 >> 1)] >> ((signed char)((k_2 & 1) * 4))) & (signed char)15))] * Scales[((((k_0 * 27648) + ((((int)threadIdx.x) >> 4) * 13824)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.y))])));
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
    
        if (M == 16 && N == 12288 && K == 4096){
            
             const dim3 GridDim(384, 1, 1);
             const dim3 BlockDim(32, 4, 1);
             cutlass_kernel_fp16_nf4_fp16_m16n12288k4096_nt_16x32x32_16x8x16_<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
        if (M == 16 && N == 4096 && K == 4096){
            
             const dim3 GridDim(256, 1, 1);
             const dim3 BlockDim(32, 1, 1);
             cutlass_kernel_fp16_nf4_fp16_m16n4096k4096_nt_8x32x32_8x32x16_<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
        if (M == 16 && N == 4096 && K == 11008){
            
             const dim3 GridDim(256, 1, 1);
             const dim3 BlockDim(32, 1, 1);
             cutlass_kernel_fp16_nf4_fp16_m16n4096k11008_nt_8x32x32_8x32x16_<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
        if (M == 16 && N == 11008 && K == 4096){
            
             const dim3 GridDim(344, 1, 1);
             const dim3 BlockDim(32, 4, 1);
             cutlass_kernel_fp16_nf4_fp16_m16n11008k4096_nt_16x32x32_16x8x16_<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
        if (M == 1 && N == 12288 && K == 4096){
            
             const dim3 GridDim(3072, 1, 1);
             const dim3 BlockDim(32, 4, 1);
             cutlass_kernel_fp16_nf4_fp16_m1n12288k4096_nt_4x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
        if (M == 1 && N == 4096 && K == 4096){
            
             const dim3 GridDim(1024, 1, 1);
             const dim3 BlockDim(32, 4, 1);
             cutlass_kernel_fp16_nf4_fp16_m1n4096k4096_nt_4x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
        if (M == 1 && N == 4096 && K == 11008){
            
             const dim3 GridDim(1024, 1, 1);
             const dim3 BlockDim(32, 4, 1);
             cutlass_kernel_fp16_nf4_fp16_m1n4096k11008_nt_4x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
        if (M == 1 && N == 11008 && K == 4096){
            
             const dim3 GridDim(2752, 1, 1);
             const dim3 BlockDim(32, 4, 1);
             cutlass_kernel_fp16_nf4_fp16_m1n11008k4096_nt_4x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
        if (M == 16 && N == 15360 && K == 5120){
            
             const dim3 GridDim(320, 1, 1);
             const dim3 BlockDim(32, 3, 1);
             cutlass_kernel_fp16_nf4_fp16_m16n15360k5120_nt_16x48x32_16x8x16_<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
        if (M == 16 && N == 5120 && K == 5120){
            
             const dim3 GridDim(320, 1, 1);
             const dim3 BlockDim(32, 1, 1);
             cutlass_kernel_fp16_nf4_fp16_m16n5120k5120_nt_8x32x32_8x32x16_<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
        if (M == 16 && N == 5120 && K == 13824){
            
             const dim3 GridDim(320, 1, 1);
             const dim3 BlockDim(32, 1, 1);
             cutlass_kernel_fp16_nf4_fp16_m16n5120k13824_nt_8x32x32_8x32x16_<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
        if (M == 16 && N == 13824 && K == 5120){
            
             const dim3 GridDim(432, 1, 1);
             const dim3 BlockDim(32, 4, 1);
             cutlass_kernel_fp16_nf4_fp16_m16n13824k5120_nt_16x32x32_16x8x16_<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
        if (M == 1 && N == 15360 && K == 5120){
            
             const dim3 GridDim(3840, 1, 1);
             const dim3 BlockDim(32, 4, 1);
             cutlass_kernel_fp16_nf4_fp16_m1n15360k5120_nt_4x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
        if (M == 1 && N == 5120 && K == 5120){
            
             const dim3 GridDim(1280, 1, 1);
             const dim3 BlockDim(32, 4, 1);
             cutlass_kernel_fp16_nf4_fp16_m1n5120k5120_nt_4x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
        if (M == 1 && N == 5120 && K == 13824){
            
             const dim3 GridDim(1280, 1, 1);
             const dim3 BlockDim(32, 4, 1);
             cutlass_kernel_fp16_nf4_fp16_m1n5120k13824_nt_4x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
        if (M == 1 && N == 13824 && K == 5120){
            
             const dim3 GridDim(3456, 1, 1);
             const dim3 BlockDim(32, 4, 1);
             cutlass_kernel_fp16_nf4_fp16_m1n13824k5120_nt_4x8<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
            return 0;
        }

        
    return -1;
}