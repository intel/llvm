set(LLVM_LINK_COMPONENTS Object Support)

if(NOT CLANG_BUILT_STANDALONE)
  set(tablegen_deps intrinsics_gen)
endif()

add_clang_tool(clang-offload-bundler
  ClangOffloadBundler.cpp
  
  DEPENDS
  ${tablegen_deps}
  )

set(CLANG_OFFLOAD_BUNDLER_LIB_DEPS
  clangBasic
  )
  
add_dependencies(clang clang-offload-bundler)

# A standalone clang build uses a pre-installed llvm-objcopy
if (NOT CLANG_BUILT_STANDALONE)
  add_dependencies(clang-offload-bundler llvm-objcopy)
endif()

clang_target_link_libraries(clang-offload-bundler
  PRIVATE
  ${CLANG_OFFLOAD_BUNDLER_LIB_DEPS}
  )
