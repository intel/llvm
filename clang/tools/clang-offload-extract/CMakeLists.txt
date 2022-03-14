set(LLVM_LINK_COMPONENTS Core Object Support)

add_clang_tool(clang-offload-extract
  ClangOffloadExtract.cpp
  CUSTOM_WIN_VER

  DEPENDS
  intrinsics_gen
  )

set(CLANG_OFFLOAD_EXTRACT_LIB_DEPS
  clangBasic
  )

add_dependencies(clang clang-offload-extract)

clang_target_link_libraries(clang-offload-extract
  PRIVATE
  ${CLANG_OFFLOAD_EXTRACT_LIB_DEPS}
  )
