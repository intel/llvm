add_mlir_conversion_library(MLIRSYCLToLLVM
  DialectBuilder.cpp
  SYCLFuncRegistry.cpp
  SYCLToLLVM.cpp
  SYCLToLLVMPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_SYCL_SOURCE_DIR}/mlir/Conversion/SYCLToLLVM

  DEPENDS
  MLIRSYCLConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMemRefToLLVM
  MLIRSYCLDialect  
  MLIRTransforms
  )
