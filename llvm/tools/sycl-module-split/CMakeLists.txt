set(LLVM_LINK_COMPONENTS
  Core
  IRReader
  Support
  SYCLPostLink
  )

add_llvm_tool(sycl-module-split
  sycl-module-split.cpp
  )
