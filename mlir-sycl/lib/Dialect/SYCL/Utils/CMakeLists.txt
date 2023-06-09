add_mlir_dialect_library(MLIRSYCLUtils
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_SYCL_SOURCE_DIR}/include/mlir/Dialect/SYCL/Utils

  LINK_LIBS PUBLIC
  MLIRSYCLDialect
  MLIRDialect
  MLIRIR
  )
