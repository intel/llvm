add_mlir_conversion_library(MLIRGPUToLLVMSPV
  GPUToLLVMSPV.cpp

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PRIVATE
  MLIRGPUDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
)
