set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

add_llvm_tool(append-file
  append-file.cpp

  DEPENDS
  intrinsics_gen
  )
