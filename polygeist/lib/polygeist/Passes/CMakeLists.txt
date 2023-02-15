add_mlir_dialect_library(MLIRPolygeistTransforms
  AffineCFG.cpp
  AffineReduction.cpp
  BarrierRemovalContinuation.cpp  
  CanonicalizeFor.cpp
  ConvertPolygeistToLLVM.cpp  
  InnerSerialization.cpp  
  LegalizeForSPIRV.cpp  
  LoopRestructure.cpp
  Mem2Reg.cpp
  OpenMPOpt.cpp
  ParallelLICM.cpp
  ParallelLoopDistribute.cpp
  ParallelLower.cpp
  RaiseToAffine.cpp
  TrivialUse.cpp

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
  MLIRVectorToLLVM
  )
