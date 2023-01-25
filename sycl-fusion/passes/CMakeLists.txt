# Module library for usage as library/pass-plugin with LLVM opt.
add_llvm_library(SYCLKernelFusion MODULE
  SYCLFusionPasses.cpp
  kernel-fusion/SYCLKernelFusion.cpp
  kernel-info/SYCLKernelInfo.cpp
  internalization/Internalization.cpp
  syclcp/SYCLCP.cpp
  cleanup/Cleanup.cpp
  debug/PassDebug.cpp

  DEPENDS
  intrinsics_gen
)

target_include_directories(SYCLKernelFusion
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE
  ${SYCL_JIT_BASE_DIR}/common/include
)

# Static library for linking with the jit_compiler
add_llvm_library(SYCLKernelFusionPasses
  SYCLFusionPasses.cpp
  kernel-fusion/SYCLKernelFusion.cpp
  kernel-info/SYCLKernelInfo.cpp
  internalization/Internalization.cpp
  syclcp/SYCLCP.cpp
  cleanup/Cleanup.cpp
  debug/PassDebug.cpp

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Core
  Support
  TransformUtils
  Passes
)

target_include_directories(SYCLKernelFusionPasses
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  PRIVATE
  ${SYCL_JIT_BASE_DIR}/common/include
)
