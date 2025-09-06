set(OCK_LIBS)
option(NATIVECPU_USE_OCK "Use the oneAPI Construction Kit for Native CPU" ON)

# Don't use OCK compiler_passes if Native CPU is not enabled.
if(NOT "native_cpu" IN_LIST SYCL_ENABLE_BACKENDS)
  set(NATIVECPU_USE_OCK Off CACHE BOOL "Use the oneAPI Construction Kit for Native CPU" FORCE)
endif()

if(NATIVECPU_USE_OCK)
  add_subdirectory(compiler_passes EXCLUDE_FROM_ALL)
  set(OCK_LIBS NativeCPUPipeline NativeCPUVecz)
endif()

add_llvm_component_library(LLVMSYCLNativeCPUUtils
  PipelineSYCLNativeCPU.cpp
  PrepareSYCLNativeCPU.cpp
  RenameKernelSYCLNativeCPU.cpp
  ConvertToMuxBuiltinsSYCLNativeCPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/SYCLLowerIR
  
  LINK_COMPONENTS
  Analysis
  Core
  Support
  Passes
  SYCLLowerIR
  Target
  TargetParser
  TransformUtils
  ipo
  ${OCK_LIBS}
)


if(NATIVECPU_USE_OCK)
  target_compile_definitions(LLVMSYCLNativeCPUUtils PRIVATE  NATIVECPU_USE_OCK)
  target_include_directories(LLVMSYCLNativeCPUUtils PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/compiler_passes/compiler_pipeline/include
    ${CMAKE_CURRENT_SOURCE_DIR}/compiler_passes/vecz/include)
endif()
