add_mlir_conversion_library(MLIRSPIRVBuiltInToNVVM
  SPIRVBuiltInToNVVM.cpp
  
  ADDITIONAL_HEADER_DIRS
  ${MLIR_SYCL_SOURCE_DIR}/mlir/Conversion/SPIRVBuiltInToNVVM
  
  DEPENDS
  MLIRSYCLConversionPassIncGen
  
  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  MLIRIR
  MLIRSYCLDialect
  MLIRTransforms
  MLIRVectorDialect
  )  