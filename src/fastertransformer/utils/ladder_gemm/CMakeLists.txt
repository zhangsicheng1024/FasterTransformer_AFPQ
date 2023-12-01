add_library(ladder_gemm STATIC 
  ladder_gemm_fp16_nf4_fp16.cu
)
set_property(TARGET ladder_gemm PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ladder_gemm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
#set_target_properties(ladder_gemm PROPERTIES INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/_cutlass/include)