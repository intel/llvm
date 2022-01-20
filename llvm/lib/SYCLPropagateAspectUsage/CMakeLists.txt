add_llvm_component_library(LLVMSYCLPropagateAspectUsage
  PropagateAspectUsage.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/SYCLPropagateAspectUsage

  DEPENDS
  LLVMDemangle
  LLVMTransformUtils

  LINK_LIBS
  LLVMDemangle
  LLVMTransformUtils

  LINK_COMPONENTS
  Analysis
  Core
  Support
)
