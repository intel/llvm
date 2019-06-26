set(LLVM_LINK_COMPONENTS
  Core
  IRReader
  Support
  )

add_llvm_tool(llvm-no-spirv-kernel
  llvm-no-spirv-kernel.cpp

  DEPENDS
  intrinsics_gen
  )
