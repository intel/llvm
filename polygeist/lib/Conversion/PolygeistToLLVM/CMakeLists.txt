add_mlir_conversion_library(MLIRPolygeistToLLVM
  PolygeistToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/PolygeistToLLVM

  DEPENDS
  MLIRPolygeistConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRAsyncDialect
  MLIRFuncTransforms
  MLIRMathToLLVM
  MLIROpenMPToLLVM
  MLIRPass
  MLIRPolygeistDialect
  MLIRPolygeistTransforms
  MLIRSCFToControlFlow
  MLIRSYCLToLLVM
  MLIRSYCLToMath
  MLIRVectorToLLVM
)
