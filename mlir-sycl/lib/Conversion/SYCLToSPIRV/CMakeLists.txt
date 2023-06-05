add_mlir_conversion_library(MLIRSYCLToSPIRV
  SYCLToSPIRV.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_SYCL_SOURCE_DIR}/mlir/Conversion/SYCLToSPIRV

  DEPENDS
  MLIRSYCLConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  MLIRIR
  MLIRSPIRVConversion
  MLIRSPIRVDialect
  MLIRSYCLDialect
  MLIRTransforms
  MLIRVectorDialect
  )
