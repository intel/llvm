set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

if(NOT CLANG_BUILT_STANDALONE)
  set(tablegen_deps intrinsics_gen)
endif()

add_clang_library(clangSYCLLowerIR
  LowerWGScope.cpp

  DEPENDS
  ${tablegen_deps}

  LINK_LIBS
  clangBasic
  )
