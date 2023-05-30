add_mlir_conversion_library(MLIRSYCLToGPU
  SYCLToGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_SYCL_SOURCE_DIR}/mlir/Conversion/SYCLToGPU

  DEPENDS
  MLIRSYCLConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  MLIRIR
  MLIRSYCLDialect
  MLIRTransforms
  MLIRVectorDialect
  )
