set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_tool(llvm-foreach
  llvm-foreach.cpp
)

setup_host_tool(llvm-foreach LLVM_FOREACH llvm-foreach_exe llvm-foreach_target)
