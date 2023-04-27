add_mlir_library(SYCLMLIRTestAnalysis
  TestReachingDefinitionsAnalysis.cpp

  EXCLUDE_FROM_LIBMLIR

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SYCL
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRIR  
  MLIRPass
  MLIRSupport  
  MLIRSYCLAnalysis  
  )
  