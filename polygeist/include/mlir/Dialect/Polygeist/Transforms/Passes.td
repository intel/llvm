//===- Passes.td - Polygeist op transform passes -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_POLYGEIST_TRANSFORMS_PASSES
#define MLIR_DIALECT_POLYGEIST_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def AffineCFG : Pass<"affine-cfg"> {
  let summary = "Replace scf.if and similar with affine.if";
  let constructor = "mlir::polygeist::replaceAffineCFGPass()";
}

def AffineReduction : Pass<"detect-reduction"> {
  let summary = "Detect reductions in affine.for";
  let constructor = "mlir::polygeist::detectReductionPass()";
  let statistics = [
    Statistic<"numReductions", "num-reductions", "Number of reductions detected">
  ];
}

def ArgumentPromotion : Pass<"arg-promotion", "mlir::ModuleOp"> {
  let summary = "Promote function arguments";
  let description = [{
    Peel struct members from the arguments passed to the function called by a SYCL kernel. 
    For example given:

      gpu.func @kernel_parallel_for(%arg0: i32) {
        func.call @parallel_for(%0, ...) : (memref<?x!llvm.struct<(i32, !sycl.accessor...)
        gpu.return
      }

    The pass "peels off" the i32 struct member:

      gpu.func @kernel_parallel_for(%arg0: i32) {
        %int_arg = <first struct member>
        %new_memref = <memref to struct with first member removed>
        func.call @parallel_for(%int_arg, %new_memref, ...) : (memref<?xi32>, memref<?x!llvm.struct<(!sycl.accessor...)
        gpu.return
      }    
  }];
  let constructor = "mlir::polygeist::createArgumentPromotionPass()";
  let statistics = [
    Statistic<"numStructMemberPeeled", "num-struct-member-peeled", "Number of struct member peeled">
  ];
}

def Mem2Reg : Pass<"mem2reg"> {
  let summary = "Represent allocated values in SSA form.";
  let constructor = "mlir::polygeist::createMem2RegPass()";
}

def ParallelLower : Pass<"parallel-lower", "mlir::ModuleOp"> {
  let summary =
    "Lower NVVM/GPU dialects into a generic parallel for representation.";
  let dependentDialects =
      ["memref::MemRefDialect", "polygeist::PolygeistDialect",
       "LLVM::LLVMDialect", "scf::SCFDialect"];
  let constructor = "mlir::polygeist::createParallelLowerPass()";
}

def SCFCPUify : Pass<"cpuify"> {
  let summary = "remove scf.barrier";
  let constructor = "mlir::polygeist::createCPUifyPass()";
  let dependentDialects = ["memref::MemRefDialect", "LLVM::LLVMDialect"];
  let options = [
  Option<"method", "method", "std::string", /*default=*/"\"distribute\"", "Method of doing distribution">
  ];
}

def InnerSerialization : Pass<"inner-serialize"> {
  let summary = "Serialize scf.parallel operations";
  let constructor = "mlir::polygeist::createInnerSerializationPass()";
  let dependentDialects = ["memref::MemRefDialect"];
}

def LLVMLegalizeForSPIRV : Pass<"legalize-for-spirv"> {
  let summary = "Legalize dialect to be convertible to SPIRV IR";
  let constructor = "::mlir::polygeist::createLegalizeForSPIRVPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];  
}

def SCFBarrierRemovalContinuation : InterfacePass<"barrier-removal-continuation", "FunctionOpInterface"> {
  let summary = "Remove scf.barrier using continuations";
  let constructor = "mlir::polygeist::createBarrierRemovalContinuation()";
  let dependentDialects = ["memref::MemRefDialect"];
}

def SCFRaiseToAffine : Pass<"raise-scf-to-affine"> {
  let summary = "Raise SCF to affine";
  let constructor = "mlir::polygeist::createRaiseSCFToAffinePass()";
  let dependentDialects = ["AffineDialect"];
}

def SCFCanonicalizeFor : Pass<"canonicalize-scf-for"> {
  let summary = "Run some additional canonicalization for scf::for";
  let constructor = "mlir::polygeist::createCanonicalizeForPass()";
  let options = [
    Option<"relaxedAliasing", "relaxed-aliasing", "bool", 
           /*default=*/"false", 
           "Whether to use strict aliasing (i.e. type-based aliasing) rules or not">
  ];
}

def LICM : Pass<"licm"> {
  let summary = "Perform LICM on known parallel (and serial) loops";
  let constructor = "mlir::polygeist::createLICMPass()";
  let options = [
    Option<"relaxedAliasing", "relaxed-aliasing", "bool", 
           /*default=*/"false", 
           "Whether to use strict aliasing (i.e. type-based aliasing) rules or not">
  ];
  let statistics = [
    Statistic<"numOpHoisted", "num-operations-hoisted", "Number of operations hoisted">
  ];
}

def OpenMPOptPass : Pass<"openmp-opt"> {
  let summary = "Optimize OpenMP";
  let constructor = "mlir::polygeist::createOpenMPOptPass()";
  let dependentDialects = ["memref::MemRefDialect"];
}

def LoopRestructure : Pass<"loop-restructure"> {
  let constructor = "mlir::polygeist::createLoopRestructurePass()";
  let dependentDialects = ["LLVM::LLVMDialect", "scf::SCFDialect"];
}

def RemoveTrivialUse : Pass<"trivialuse"> {
  let constructor = "mlir::polygeist::createRemoveTrivialUsePass()";
}

#endif // MLIR_DIALECT_POLYGEIST_TRANSFORMS_PASSES
