set(LLVM_LINK_COMPONENTS
  BinaryFormat
  BitReader
  BitWriter
  Core
  IRReader
  Linker
  Object
  Support
  TransformUtils
  IPO
  )

add_llvm_tool(llvm-link
  llvm-link.cpp
  CUSTOM_WIN_VER

  DEPENDS
  intrinsics_gen
  )
