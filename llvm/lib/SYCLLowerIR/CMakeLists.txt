add_llvm_component_library(LLVMSYCLLowerIR
  LowerWGScope.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/SYCLLowerIR

  DEPENDS
  intrinsics_gen
  )
