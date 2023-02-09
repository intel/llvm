#ifndef MLIR_DIALECT_POLYGEIST_TRANSFORMS_PASSES
#define MLIR_DIALECT_POLYGEIST_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def AffineCFG : Pass<"affine-cfg"> {
  let summary = "Replace scf.if and similar with affine.if";
  let constructor = "mlir::polygeist::replaceAffineCFGPass()";
}

def Mem2Reg : Pass<"mem2reg"> {
  let summary = "Replace scf.if and similar with affine.if";
  let constructor = "mlir::polygeist::createMem2RegPass()";
}

def ParallelLower : Pass<"parallel-lower", "mlir::ModuleOp"> {
  let summary = "Replace scf.if and similar with affine.if";
  let dependentDialects =
      ["memref::MemRefDialect", "func::FuncDialect", "LLVM::LLVMDialect"];
  let constructor = "mlir::polygeist::createParallelLowerPass()";
}

def AffineReduction : Pass<"detect-reduction"> {
  let summary = "Detect reductions in affine.for";
  let constructor = "mlir::polygeist::detectReductionPass()";
}

def SCFCPUify : Pass<"cpuify"> {
  let summary = "remove scf.barrier";
  let constructor = "mlir::polygeist::createCPUifyPass()";
  let dependentDialects =
      ["memref::MemRefDialect", "func::FuncDialect", "LLVM::LLVMDialect"];
  let options = [
  Option<"method", "method", "std::string", /*default=*/"\"distribute\"", "Method of doing distribution">
  ];
}

def InnerSerialization : Pass<"inner-serialize"> {
  let summary = "remove scf.barrier";
  let constructor = "mlir::polygeist::createInnerSerializationPass()";
  let dependentDialects =
      ["memref::MemRefDialect", "func::FuncDialect", "LLVM::LLVMDialect"];
}

def LLVMLegalizeForSPIRV : Pass<"legalize-for-spirv"> {
  let summary = "Legalize dialect to be convertible to SPIRV IR";
  let constructor = "::mlir::polygeist::createLegalizeForSPIRVPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];  
}

def SCFBarrierRemovalContinuation : InterfacePass<"barrier-removal-continuation", "FunctionOpInterface"> {
  let summary = "Remove scf.barrier using continuations";
  let constructor = "mlir::polygeist::createBarrierRemovalContinuation()";
  let dependentDialects = ["memref::MemRefDialect", "func::FuncDialect"];
}

def SCFRaiseToAffine : Pass<"raise-scf-to-affine"> {
  let summary = "Raise SCF to affine";
  let constructor = "mlir::polygeist::createRaiseSCFToAffinePass()";
  let dependentDialects = ["AffineDialect"];
}

def SCFCanonicalizeFor : Pass<"canonicalize-scf-for"> {
  let summary = "Run some additional canonicalization for scf::for";
  let constructor = "mlir::polygeist::createCanonicalizeForPass()";
}

def ParallelLICM : Pass<"parallel-licm"> {
  let summary = "Perform LICM on known parallel (and serial) loops";
  let constructor = "mlir::polygeist::createParallelLICMPass()";
}

def OpenMPOptPass : Pass<"openmp-opt"> {
  let summary = "Optimize OpenMP";
  let constructor = "mlir::polygeist::createOpenMPOptPass()";
}

def LoopRestructure : Pass<"loop-restructure"> {
  let constructor = "mlir::polygeist::createLoopRestructurePass()";
  let dependentDialects = ["::mlir::scf::SCFDialect"];
}

def RemoveTrivialUse : Pass<"trivialuse"> {
  let constructor = "mlir::polygeist::createRemoveTrivialUsePass()";
}

#endif // MLIR_DIALECT_POLYGEIST_TRANSFORMS_PASSES
