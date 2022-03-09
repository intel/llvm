set(LLVM_LINK_COMPONENTS BitWriter Core Object Support TransformUtils)

add_clang_tool(clang-offload-wrapper
  ClangOffloadWrapper.cpp

  DEPENDS
  intrinsics_gen
  CUSTOM_WIN_VER
  )

set(CLANG_OFFLOAD_WRAPPER_LIB_DEPS
  clangBasic
  )

add_dependencies(clang clang-offload-wrapper)

clang_target_link_libraries(clang-offload-wrapper
  PRIVATE
  ${CLANG_OFFLOAD_WRAPPER_LIB_DEPS}
  )
