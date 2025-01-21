set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsDescs
  AllTargetsInfos
  BitWriter
  Core
  IRReader
  BinaryFormat
  Object
  Support
  TargetParser
  )

add_clang_tool(clang-offload-bundler
  ClangOffloadBundler.cpp

  DEPENDS
  intrinsics_gen
  )

setup_host_tool(clang-offload-bundler CLANG_OFFLOAD_BUNDLER
  clang-offload-bundler_exe clang-offload-bundler_target)

set(CLANG_OFFLOAD_BUNDLER_LIB_DEPS
  clangBasic
  clangDriver
  )

# A standalone clang build uses a pre-installed llvm-objcopy
if (NOT CLANG_BUILT_STANDALONE)
  add_dependencies(clang-offload-bundler llvm-objcopy)
endif()

clang_target_link_libraries(clang-offload-bundler
  PRIVATE
  ${CLANG_OFFLOAD_BUNDLER_LIB_DEPS}
  )
