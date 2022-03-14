set(LLVM_LINK_COMPONENTS
  AllTargetsInfos
  MC
  Object
  Support
  )

add_llvm_tool(llvm-cxxdump
  llvm-cxxdump.cpp
  Error.cpp
  CUSTOM_WIN_VER
  )
