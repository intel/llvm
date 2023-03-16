add_mlir_dialect_library(MLIRPolygeistTransforms
  AffineCFG.cpp
  AffineReduction.cpp
  ArgumentPromotion.cpp  
  BareMemRefToLLVM.cpp
  BarrierRemovalContinuation.cpp  
  CanonicalizeFor.cpp
  InnerSerialization.cpp  
  LegalizeForSPIRV.cpp  
  LICM.cpp
  LoopRestructure.cpp
  Mem2Reg.cpp
  OpenMPOpt.cpp
  ParallelLoopDistribute.cpp
  ParallelLower.cpp
  RaiseToAffine.cpp
  TrivialUse.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Polygeist

  DEPENDS
  MLIRPolygeistOpsIncGen
  MLIRPolygeistPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAsyncDialect
  MLIRAffineUtils
  MLIRFuncDialect
  MLIRGPUOps
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRNVVMDialect
  MLIRPass
  MLIRPolygeist
  MLIRSYCLAnalysis  
  MLIRSideEffectInterfaces
  MLIRSCFToControlFlow
  MLIRTransformUtils
)
