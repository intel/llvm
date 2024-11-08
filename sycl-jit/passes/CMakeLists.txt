# See llvm/examples/Bye/CmakeLists.txt as to why this kind of loadable plugin libraries
# isn't supported on Windows.
if (NOT WIN32 AND NOT CYGWIN)
  # Module library for usage as library/pass-plugin with LLVM opt.
  add_llvm_library(SYCLKernelJIT MODULE
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

  target_compile_options(SYCLKernelJIT PRIVATE ${SYCL_JIT_WARNING_FLAGS})

  # Mark LLVM headers as system headers to ignore warnigns in them. This
  # classification remains intact even if the same path is added as a normal
  # include path in GCC and Clang.
  target_include_directories(SYCLKernelJIT
    SYSTEM PRIVATE
    ${LLVM_MAIN_INCLUDE_DIR}
  )
  target_include_directories(SYCLKernelJIT
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
    ${SYCL_JIT_BASE_DIR}/common/include
  )

  target_link_libraries(SYCLKernelJIT
    PRIVATE
    sycl-jit-common
  )

  add_dependencies(SYCLKernelJIT sycl-headers)

  if("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
    target_compile_definitions(SYCLKernelJIT PRIVATE JIT_SUPPORT_PTX)
  endif()

  if("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
    target_compile_definitions(SYCLKernelJIT PRIVATE JIT_SUPPORT_AMDGCN)
  endif()

endif()

# Static library for linking with the jit_compiler
add_llvm_library(SYCLKernelJITPasses
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
  SYCLLowerIR
)

target_compile_options(SYCLKernelJITPasses PRIVATE ${SYCL_JIT_WARNING_FLAGS})

# Mark LLVM headers as system headers to ignore warnigns in them. This
# classification remains intact even if the same path is added as a normal
# include path in GCC and Clang.
target_include_directories(SYCLKernelJITPasses
  SYSTEM PRIVATE
  ${LLVM_MAIN_INCLUDE_DIR}
)
target_include_directories(SYCLKernelJITPasses
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  PRIVATE
  ${SYCL_JIT_BASE_DIR}/common/include
)

target_link_libraries(SYCLKernelJITPasses
  PRIVATE
  sycl-jit-common
)

add_dependencies(SYCLKernelJITPasses sycl-headers)

if("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
  target_compile_definitions(SYCLKernelJITPasses PRIVATE JIT_SUPPORT_PTX)
endif()

if("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  target_compile_definitions(SYCLKernelJITPasses PRIVATE JIT_SUPPORT_AMDGCN)
endif()
