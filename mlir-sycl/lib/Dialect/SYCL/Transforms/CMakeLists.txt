add_mlir_library(MLIRSYCLTransforms
  CPP.cpp
  Inliner.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  DEPENDS
  MLIRSYCLTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransforms  
  MLIRTransformUtils
  MLIRSYCLDialect
  MLIRFuncDialect
  MLIRGPUDialect
  )
