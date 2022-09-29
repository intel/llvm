add_mlir_dialect_library(MLIRPolygeistTransforms
  AffineCFG.cpp
  AffineReduction.cpp
  CanonicalizeFor.cpp
  LoopRestructure.cpp
  Mem2Reg.cpp
  ParallelLoopDistribute.cpp
  ParallelLICM.cpp
  OpenMPOpt.cpp
  BarrierRemovalContinuation.cpp
  RaiseToAffine.cpp
  ParallelLower.cpp
  TrivialUse.cpp
  ConvertPolygeistToLLVM.cpp
  InnerSerialization.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Affine

  DEPENDS
  MLIRPolygeistOpsIncGen
  MLIRPolygeistPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAsyncDialect
  MLIRAffineUtils
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRGPUOps
  MLIRIR
  MLIRMathDialect
  MLIRMathToLLVM
  MLIRMemRefDialect
  MLIRNVVMDialect
  MLIROpenMPToLLVM
  MLIRPass
  MLIRPolygeist
  MLIRSideEffectInterfaces
  MLIRSCFToControlFlow
  MLIRSYCLDialect
  MLIRSYCLToLLVM
  MLIRTransformUtils
  )
