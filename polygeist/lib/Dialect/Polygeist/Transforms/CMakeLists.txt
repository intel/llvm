add_mlir_dialect_library(MLIRPolygeistTransforms
  AffineCFG.cpp
  ArgumentPromotion.cpp  
  BareMemRefToLLVM.cpp
  BarrierRemovalContinuation.cpp  
  CanonicalizeFor.cpp
  DetectReduction.cpp
  InnerSerialization.cpp  
  KernelDisjointSpecialization.cpp
  LegalizeForSPIRV.cpp  
  LICM.cpp
  LoopInternalization.cpp
  LoopRestructure.cpp
  Mem2Reg.cpp
  OpenMPOpt.cpp
  ParallelLoopDistribute.cpp
  ParallelLower.cpp
  RaiseToAffine.cpp
  SYCLRaiseHost.cpp
  TrivialUse.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Polygeist/Transforms

  DEPENDS
  MLIRPolygeistOpsIncGen
  MLIRPolygeistPassIncGen

  LINK_COMPONENTS
  Demangle

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAffineToStandard
  MLIRAsyncDialect
  MLIRAffineUtils
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRMemRefToLLVM
  MLIRNVVMDialect
  MLIRPass
  MLIRPolygeistAnalysis  
  MLIRPolygeistDialect
  MLIRPolygeistUtils
  MLIRSCFUtils
  MLIRSPIRVDialect
  MLIRSYCLAnalysis  
  MLIRSYCLDialect
  MLIRSYCLUtils
  MLIRSideEffectInterfaces
  MLIRSCFToControlFlow
  MLIRTransformUtils
  MLIRVectorDialect
)
