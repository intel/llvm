macro(configure_in_llvm_tree)
  set(LLVM_CLANG ${LLVM_RUNTIME_OUTPUT_INTDIR}/clang)
  set(LLVM_AS ${LLVM_RUNTIME_OUTPUT_INTDIR}/llvm-as)
  set(LLVM_LINK ${LLVM_RUNTIME_OUTPUT_INTDIR}/llvm-link)
  set(LLVM_OPT ${LLVM_RUNTIME_OUTPUT_INTDIR}/opt)

  if (NOT EXISTS ${LLVM_RUNTIME_OUTPUT_INTDIR}/clang)
    file(WRITE ${LLVM_RUNTIME_OUTPUT_INTDIR}/clang "" )
  endif (NOT EXISTS ${LLVM_RUNTIME_OUTPUT_INTDIR}/clang)
  if (NOT EXISTS ${LLVM_RUNTIME_OUTPUT_INTDIR}/llvm-as)
    file(WRITE ${LLVM_RUNTIME_OUTPUT_INTDIR}/llvm-as "" )
  endif (NOT EXISTS ${LLVM_RUNTIME_OUTPUT_INTDIR}/llvm-as)
  if (NOT EXISTS ${LLVM_RUNTIME_OUTPUT_INTDIR}/llvm-link)
    file(WRITE ${LLVM_RUNTIME_OUTPUT_INTDIR}/llvm-link "" )
  endif (NOT EXISTS ${LLVM_RUNTIME_OUTPUT_INTDIR}/llvm-link)
  if (NOT EXISTS ${LLVM_RUNTIME_OUTPUT_INTDIR}/opt)
    file(WRITE ${LLVM_RUNTIME_OUTPUT_INTDIR}/opt "" )
  endif (NOT EXISTS ${LLVM_RUNTIME_OUTPUT_INTDIR}/opt)

  # Assume all works well
  # We can't test the compilers as they haven't been built yet
  set(CMAKE_CLC_COMPILER_FORCED TRUE)
  set(CMAKE_LLAsm_COMPILER_FORCED TRUE)
endmacro(configure_in_llvm_tree)

configure_in_llvm_tree()
