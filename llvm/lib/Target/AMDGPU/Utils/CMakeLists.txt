add_llvm_component_library(LLVMAMDGPUUtils
  AMDGPUAsmUtils.cpp
  AMDGPUBaseInfo.cpp
  AMDGPUDelayedMCExpr.cpp
  AMDGPUPALMetadata.cpp
  AMDKernelCodeTUtils.cpp

  LINK_COMPONENTS
  Analysis
  BinaryFormat
  CodeGenTypes
  Core
  MC
  MCParser
  Support
  TargetParser
  TransformUtils

  ADD_TO_COMPONENT
  AMDGPU
  )
