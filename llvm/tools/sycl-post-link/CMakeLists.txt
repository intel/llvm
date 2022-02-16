set(LLVM_LINK_COMPONENTS
  BitWriter
  Core
  IPO
  IRReader
  Support
  TransformUtils
  SYCLLowerIR
  InstCombine
  ScalarOpts
  )

get_property(LLVMGenXIntrinsics_SOURCE_DIR GLOBAL PROPERTY LLVMGenXIntrinsics_SOURCE_PROP)
get_property(LLVMGenXIntrinsics_BINARY_DIR GLOBAL PROPERTY LLVMGenXIntrinsics_BINARY_PROP)

include_directories(
  ${LLVMGenXIntrinsics_SOURCE_DIR}/GenXIntrinsics/include
  ${LLVMGenXIntrinsics_BINARY_DIR}/GenXIntrinsics/include)

add_llvm_tool(sycl-post-link
  sycl-post-link.cpp
  CompileTimePropertiesPass.cpp
  DeviceGlobals.cpp
  SpecConstants.cpp
  SYCLDeviceLibReqMask.cpp
  SYCLKernelParamOptInfo.cpp
  ADDITIONAL_HEADER_DIRS
  ${LLVMGenXIntrinsics_SOURCE_DIR}/GenXIntrinsics/include
  ${LLVMGenXIntrinsics_BINARY_DIR}/GenXIntrinsics/include

  DEPENDS
  intrinsics_gen
  LLVMGenXIntrinsics
  )

target_link_libraries(sycl-post-link PRIVATE LLVMGenXIntrinsics)
