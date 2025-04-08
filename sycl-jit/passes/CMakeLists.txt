# See llvm/examples/Bye/CmakeLists.txt as to why this kind of loadable plugin libraries
# isn't supported on Windows.
if (NOT WIN32 AND NOT CYGWIN)
  # Module library for usage as library/pass-plugin with LLVM opt.
  add_llvm_library(SYCLJITPassesPlugin MODULE
    SYCLJITPasses.cpp
    materializer/SYCLSpecConstMaterializer.cpp

    DEPENDS
    intrinsics_gen
  )

  target_compile_options(SYCLJITPassesPlugin PRIVATE ${SYCL_JIT_WARNING_FLAGS})

  # Mark LLVM headers as system headers to ignore warnigns in them. This
  # classification remains intact even if the same path is added as a normal
  # include path in GCC and Clang.
  target_include_directories(SYCLJITPassesPlugin
    SYSTEM PRIVATE
    ${LLVM_MAIN_INCLUDE_DIR}
  )
  target_include_directories(SYCLJITPassesPlugin
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
  )

  add_dependencies(SYCLJITPassesPlugin sycl-headers)

  if("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
    target_compile_definitions(SYCLJITPassesPlugin PRIVATE JIT_SUPPORT_PTX)
  endif()

  if("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
    target_compile_definitions(SYCLJITPassesPlugin PRIVATE JIT_SUPPORT_AMDGCN)
  endif()

endif()

# Static library for linking with the jit_compiler
add_llvm_library(SYCLJITPasses
  SYCLJITPasses.cpp
  materializer/SYCLSpecConstMaterializer.cpp

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

target_compile_options(SYCLJITPasses PRIVATE ${SYCL_JIT_WARNING_FLAGS})

# Mark LLVM headers as system headers to ignore warnigns in them. This
# classification remains intact even if the same path is added as a normal
# include path in GCC and Clang.
target_include_directories(SYCLJITPasses
  SYSTEM PRIVATE
  ${LLVM_MAIN_INCLUDE_DIR}
)
target_include_directories(SYCLJITPasses
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

add_dependencies(SYCLJITPasses sycl-headers)

if("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
  target_compile_definitions(SYCLJITPasses PRIVATE JIT_SUPPORT_PTX)
endif()

if("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  target_compile_definitions(SYCLJITPasses PRIVATE JIT_SUPPORT_AMDGCN)
endif()
