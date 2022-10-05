#ifndef POlYGEIST_PASSES
#define POLYGEIST_PASSES

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

def ConvertPolygeistToLLVM : Pass<"convert-polygeist-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert scalar and vector operations from the Standard to the "
                "LLVM dialect";
  let description = [{
    Convert standard operations into the LLVM IR dialect operations.

    #### Input invariant

    -   operations including: arithmetic on integers and floats, constants,
        direct calls, returns and branches;
    -   no `tensor` types;
    -   all `vector` are one-dimensional;
    -   all blocks are reachable by following the successors of the first basic
        block;

    If other operations are present and their results are required by the LLVM
    IR dialect operations, the pass will fail.  Any LLVM IR operations or types
    already present in the IR will be kept as is.

    #### Output IR

    Functions converted to LLVM IR. Function arguments types are converted
    one-to-one. Function results are converted one-to-one and, in case more than
    1 value is returned, packed into an LLVM IR struct type. Function calls and
    returns are updated accordingly. Block argument types are updated to use
    LLVM IR types.
  }];
  let constructor = "mlir::polygeist::createConvertPolygeistToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
  let options = [
    Option<"useBarePtrCallConv", "use-bare-ptr-memref-call-conv", "bool",
           /*default=*/"false",
           "Replace FuncOp's MemRef arguments with bare pointers to the MemRef "
           "element types">,
    Option<"emitCWrappers", "emit-c-wrappers", "bool", /*default=*/"false",
           "Emit wrappers for C-compatible pointer-to-struct memref "
           "descriptors">,
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
    Option<"dataLayout", "data-layout", "std::string",
           /*default=*/"\"\"",
           "String description (LLVM format) of the data layout that is "
           "expected on the produced module">
  ];
}

#endif // POLYGEIST_PASSES
