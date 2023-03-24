add_mlir_conversion_library(MLIRSYCLToLLVM
  DialectBuilder.cpp
  SYCLFuncRegistry.cpp
  SYCLToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_SYCL_SOURCE_DIR}/mlir/Conversion/SYCLToLLVM

  DEPENDS
  MLIRSYCLConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRFuncTransforms
  MLIRGPUToSPIRV
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMemRefToLLVM
  MLIRPolygeistTransforms
  MLIRReconcileUnrealizedCasts
  MLIRSPIRVToLLVM
  MLIRSYCLDialect  
  MLIRSYCLToSPIRV
  MLIRTransforms
  MLIRVectorToLLVM
  )
