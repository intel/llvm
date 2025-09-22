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

setup_host_tool(sycl-post-link SYCL_POST_LINK
  sycl-post-link_exe sycl-post-link_target)
