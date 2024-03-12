# Module library for usage as library/pass-plugin with LLVM opt.
add_library(SYCLKernelFusion MODULE
  SYCLFusionPasses.cpp
  kernel-fusion/Builtins.cpp
  kernel-fusion/SYCLKernelFusion.cpp
  kernel-info/SYCLKernelInfo.cpp
  internalization/Internalization.cpp
  syclcp/SYCLCP.cpp
  cleanup/Cleanup.cpp
  debug/PassDebug.cpp
  target/TargetFusionInfo.cpp
)

add_dependencies(SYCLKernelFusion intrinsics_gen)

target_compile_options(SYCLKernelFusion PRIVATE -fno-rtti ${SYCL_FUSION_WARNING_FLAGS})

set_target_properties(SYCLKernelFusion PROPERTIES PREFIX "")

target_include_directories(SYCLKernelFusion
  SYSTEM PRIVATE
  ${LLVM_MAIN_INCLUDE_DIR}
)
target_include_directories(SYCLKernelFusion
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE
  ${SYCL_JIT_BASE_DIR}/common/include
)

target_link_libraries(SYCLKernelFusion
  PRIVATE
  sycl-fusion-common
)

if("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
  target_compile_definitions(SYCLKernelFusion PRIVATE FUSION_JIT_SUPPORT_PTX)
endif()

if("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  target_compile_definitions(SYCLKernelFusion PRIVATE FUSION_JIT_SUPPORT_AMDGCN)
endif()

# Static library for linking with the jit_compiler
add_library(SYCLKernelFusionPasses
  SYCLFusionPasses.cpp
  kernel-fusion/Builtins.cpp
  kernel-fusion/SYCLKernelFusion.cpp
  kernel-info/SYCLKernelInfo.cpp
  internalization/Internalization.cpp
  syclcp/SYCLCP.cpp
  cleanup/Cleanup.cpp
  debug/PassDebug.cpp
  target/TargetFusionInfo.cpp
)

add_dependencies(SYCLKernelFusionPasses intrinsics_gen)

llvm_map_components_to_libnames(llvm_libs
  Core
  Support
  TransformUtils
  Passes
  TargetParser
)

target_compile_options(SYCLKernelFusionPasses PRIVATE -fno-rtti ${SYCL_FUSION_WARNING_FLAGS})

target_include_directories(SYCLKernelFusionPasses
  SYSTEM PRIVATE
  ${LLVM_MAIN_INCLUDE_DIR}
)
target_include_directories(SYCLKernelFusionPasses
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  PRIVATE
  ${SYCL_JIT_BASE_DIR}/common/include
)

target_link_libraries(SYCLKernelFusionPasses
  PRIVATE
  sycl-fusion-common
  ${llvm_libs}
)

if("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
  target_compile_definitions(SYCLKernelFusionPasses PRIVATE FUSION_JIT_SUPPORT_PTX)
endif()

if("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  target_compile_definitions(SYCLKernelFusionPasses PRIVATE FUSION_JIT_SUPPORT_AMDGCN)
endif()
