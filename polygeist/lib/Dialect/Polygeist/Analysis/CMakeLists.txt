add_mlir_dialect_library(MLIRPolygeistAnalysis
  MemoryAccessAnalysis.cpp
  ReachingDefinitionAnalysis.cpp
  UniformityAnalysis.cpp
  
  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Polygeist/Analysis

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRAnalysis
  MLIRPolygeistUtils
  MLIRAffineAnalysis  
)

