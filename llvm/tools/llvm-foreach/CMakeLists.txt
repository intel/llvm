set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_tool(llvm-foreach
  llvm-foreach.cpp
  CUSTOM_WIN_VER
)
