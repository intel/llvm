add_mlir_conversion_library(MLIRSYCLToMath
  SYCLToMath.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_SYCL_SOURCE_DIR}/mlir/Conversion/SYCLToMath

  DEPENDS
  MLIRSYCLConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMathDialect
  MLIRSYCLDialect
  MLIRTransforms
  )
