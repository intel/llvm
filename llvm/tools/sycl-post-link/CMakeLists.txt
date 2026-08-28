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
  SYCLPostLink
  Passes
  Analysis
  )

add_llvm_tool(sycl-post-link
  sycl-post-link.cpp

  DEPENDS
  intrinsics_gen
  )

# sycl-post-link.cpp includes SYCLLowerIR's ESIMDUtils.h, which pulls in a
# vc-intrinsics header (GenXMetadata.h). In LLVM_LINK_LLVM_DYLIB builds this
# tool links libLLVM.so rather than the LLVMSYCLLowerIR component, so it does
# not inherit the LLVMGenXIntrinsics INTERFACE include directories. Add them
# explicitly so the header is found.
if (LLVM_LINK_LLVM_DYLIB)
  target_include_directories(sycl-post-link PRIVATE
    $<TARGET_PROPERTY:LLVMGenXIntrinsics,INTERFACE_INCLUDE_DIRECTORIES>)
endif()

setup_host_tool(sycl-post-link SYCL_POST_LINK
  sycl-post-link_exe sycl-post-link_target)
