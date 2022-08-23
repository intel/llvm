add_mlir_conversion_library(MLIRSYCLToLLVM
  SYCLToLLVM.cpp
  SYCLToLLVMPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_SYCL_SOURCE_DIR}/mlir/Conversion/SYCLToLLVM

  DEPENDS
  MLIRSYCLConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithmeticToLLVM
  MLIRFuncToLLVM
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMemRefToLLVM
  MLIRSYCLDialect  
  MLIRTransforms
  )
