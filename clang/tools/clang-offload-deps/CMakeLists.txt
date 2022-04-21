set(LLVM_LINK_COMPONENTS BitWriter Core Object Support)

add_clang_tool(clang-offload-deps
  ClangOffloadDeps.cpp

  DEPENDS
  intrinsics_gen
  )

set(CLANG_OFFLOAD_DEPS_LIB_DEPS
  clangBasic
  )

add_dependencies(clang clang-offload-deps)

clang_target_link_libraries(clang-offload-deps
  PRIVATE
  ${CLANG_OFFLOAD_DEPS_LIB_DEPS}
  )
