add_mlir_conversion_library(MLIRGENToLLVM
  GENToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/GENToLLVM

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRGENDialect  
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
)
