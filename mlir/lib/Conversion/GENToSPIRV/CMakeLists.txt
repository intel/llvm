add_mlir_conversion_library(MLIRGENToSPIRV
  GENToSPIRV.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/GENToSPIRV

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRGENDialect
  MLIRSPIRVConversion
  MLIRSPIRVDialect
  )
