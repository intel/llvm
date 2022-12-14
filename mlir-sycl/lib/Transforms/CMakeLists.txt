add_mlir_library(MLIRSYCLTransforms
  Inliner.cpp
  SYCLMethodToSYCLCall.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms
  ${POLYGEIST_INCLUDE_DIR}

  DEPENDS
  MLIRSYCLTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransforms  
  MLIRTransformUtils
  MLIRSYCLDialect
  MLIRFuncDialect
  MLIRGPUOps
  MLIRPolygeist
  )
