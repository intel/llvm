add_mlir_dialect_library(MLIRPolygeistUtils
  TransformUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Polygeist/Utils

  LINK_LIBS PUBLIC
  MLIRGPUOps
  MLIRPolygeistDialect
  MLIRSYCLDialect
  MLIRDialect
  MLIRIR
  )
