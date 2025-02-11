set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_tool(file-table-tform
  file-table-tform.cpp
  )

setup_host_tool(file-table-tform FILE_TABLE_TFORM file-table-tform_exe file-table-tform_target)
