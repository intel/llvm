set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

add_llvm_tool(append-file
  append-file.cpp
  CUSTOM_WIN_VER

  DEPENDS
  intrinsics_gen
  )
