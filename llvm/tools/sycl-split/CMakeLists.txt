set(LLVM_LINK_COMPONENTS
  BitWriter
  Core
  IPO
  IRReader
  Support
  )

add_llvm_tool(sycl-split
  sycl-split.cpp

  DEPENDS
  intrinsics_gen
  )
