add_mlir_conversion_library(MLIRSYCLToSPIRV
  SYCLToSPIRV.cpp
  SYCLToSPIRVPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_SYCL_SOURCE_DIR}/mlir/Conversion/SYCLToSPIRV

  DEPENDS
  MLIRSYCLConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRGPUOps
  MLIRIR
  MLIRSPIRVConversion
  MLIRSPIRVDialect
  MLIRSYCLDialect
  MLIRTransforms
  MLIRVectorDialect
  )
