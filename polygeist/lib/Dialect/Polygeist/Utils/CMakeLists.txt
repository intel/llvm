add_mlir_dialect_library(MLIRPolygeistUtils
  AliasUtils.cpp
  TransformUtils.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Polygeist/Utils

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRDialect  
  MLIRGPUDialect
  MLIRIR  
  MLIRPolygeistDialect
  MLIRSYCLDialect
  MLIRSYCLUtils
  )
