add_llvm_component_library(LLVMSYCLPostLink
  ComputeModuleRuntimeInfo.cpp
  ESIMDPostSplitProcessing.cpp
  ModuleSplitter.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/SYCLPostLink

  DEPENDS
  intrinsics_gen
  LLVMDemangle
  LLVMTransformUtils

  LINK_LIBS
  LLVMDemangle
  LLVMTargetParser
  LLVMTransformUtils
  LLVMSYCLLowerIR

  LINK_COMPONENTS
  Analysis
  BitWriter
  Core
  Demangle
  IRPrinter
  Linker
  Passes
  Support
  ipo
  )
