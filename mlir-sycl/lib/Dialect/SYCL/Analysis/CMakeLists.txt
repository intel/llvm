add_mlir_dialect_library(MLIRSYCLAnalysis
  AliasAnalysis.cpp
  MemoryAccessAnalysis.cpp  
  ReachingDefinitionAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_SYCL_INCLUDE_DIR}/mlir/Analysis

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRSYCLDialect
  )
