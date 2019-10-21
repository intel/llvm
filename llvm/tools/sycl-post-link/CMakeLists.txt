set(LLVM_LINK_COMPONENTS
  BitWriter
  Core
  IPO
  IRReader
  Support
  TransformUtils
  )

add_llvm_tool(sycl-post-link
  sycl-post-link.cpp

  DEPENDS
  intrinsics_gen
  )
