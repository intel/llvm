set(LLVM_LINK_COMPONENTS
  BitWriter
  Core
  IPO
  IRPrinter
  IRReader
  Support
  TargetParser
  TransformUtils
  SYCLLowerIR
  InstCombine
  ScalarOpts
  Linker
  Passes
  Analysis
  )

get_property(LLVMGenXIntrinsics_SOURCE_DIR GLOBAL PROPERTY LLVMGenXIntrinsics_SOURCE_PROP)
get_property(LLVMGenXIntrinsics_BINARY_DIR GLOBAL PROPERTY LLVMGenXIntrinsics_BINARY_PROP)

include_directories(
  ${LLVMGenXIntrinsics_SOURCE_DIR}/GenXIntrinsics/include
  ${LLVMGenXIntrinsics_BINARY_DIR}/GenXIntrinsics/include)

add_llvm_tool(sycl-post-link
  sycl-post-link.cpp
  ADDITIONAL_HEADER_DIRS
  ${LLVMGenXIntrinsics_SOURCE_DIR}/GenXIntrinsics/include
  ${LLVMGenXIntrinsics_BINARY_DIR}/GenXIntrinsics/include

  DEPENDS
  intrinsics_gen
  LLVMGenXIntrinsics
  )

target_link_libraries(sycl-post-link PRIVATE LLVMGenXIntrinsics)
