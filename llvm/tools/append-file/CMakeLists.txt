set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

add_llvm_tool(append-file
  append-file.cpp

  DEPENDS
  intrinsics_gen
  )

setup_host_tool(append-file APPEND_FILE append_file_exe append_file_target)
