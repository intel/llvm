# Module library for usage as library/pass-plugin with LLVM opt.
add_llvm_library(SYCLKernelJit MODULE
  SYCLFusionPasses.cpp
  kernel-fusion/Builtins.cpp
  kernel-fusion/SYCLKernelFusion.cpp
  kernel-fusion/SYCLSpecConstMaterializer.cpp
  kernel-info/SYCLKernelInfo.cpp
  internalization/Internalization.cpp
  syclcp/SYCLCP.cpp
  cleanup/Cleanup.cpp
  debug/PassDebug.cpp
  target/TargetFusionInfo.cpp

  DEPENDS
  intrinsics_gen
)

target_compile_options(SYCLKernelJit PRIVATE ${SYCL_FUSION_WARNING_FLAGS})

# Mark LLVM headers as system headers to ignore warnigns in them. This
# classification remains intact even if the same path is added as a normal
# include path in GCC and Clang.
target_include_directories(SYCLKernelJit
  SYSTEM PRIVATE
  ${LLVM_MAIN_INCLUDE_DIR}
)
target_include_directories(SYCLKernelJit
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE
  ${SYCL_JIT_BASE_DIR}/common/include
)

target_link_libraries(SYCLKernelJit
  PRIVATE
  sycl-jit-common
)

add_dependencies(SYCLKernelJit sycl-headers)

if("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
  target_compile_definitions(SYCLKernelJit PRIVATE FUSION_JIT_SUPPORT_PTX)
endif()

if("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  target_compile_definitions(SYCLKernelJit PRIVATE FUSION_JIT_SUPPORT_AMDGCN)
endif()

# Static library for linking with the jit_compiler
add_llvm_library(SYCLKernelJitPasses
  SYCLFusionPasses.cpp
  kernel-fusion/Builtins.cpp
  kernel-fusion/SYCLKernelFusion.cpp
  kernel-fusion/SYCLSpecConstMaterializer.cpp
  kernel-info/SYCLKernelInfo.cpp
  internalization/Internalization.cpp
  syclcp/SYCLCP.cpp
  cleanup/Cleanup.cpp
  debug/PassDebug.cpp
  target/TargetFusionInfo.cpp

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Core
  Support
  TransformUtils
  Passes
  TargetParser
)

target_compile_options(SYCLKernelJitPasses PRIVATE ${SYCL_FUSION_WARNING_FLAGS})

# Mark LLVM headers as system headers to ignore warnigns in them. This
# classification remains intact even if the same path is added as a normal
# include path in GCC and Clang.
target_include_directories(SYCLKernelJitPasses
  SYSTEM PRIVATE
  ${LLVM_MAIN_INCLUDE_DIR}
)
target_include_directories(SYCLKernelJitPasses
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  PRIVATE
  ${SYCL_JIT_BASE_DIR}/common/include
)

target_link_libraries(SYCLKernelJitPasses
  PRIVATE
  sycl-jit-common
)

add_dependencies(SYCLKernelJitPasses sycl-headers)

if("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
  target_compile_definitions(SYCLKernelJitPasses PRIVATE FUSION_JIT_SUPPORT_PTX)
endif()

if("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  target_compile_definitions(SYCLKernelJitPasses PRIVATE FUSION_JIT_SUPPORT_AMDGCN)
endif()
