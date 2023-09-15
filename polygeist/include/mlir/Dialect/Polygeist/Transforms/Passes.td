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
include "mlir/Rewrite/PassUtil.td"

def AffineCFG : Pass<"affine-cfg"> {
  let summary = "Replace scf.if and similar with affine.if";
  let constructor = "mlir::polygeist::createReplaceAffineCFGPass()";
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

def DetectReduction : Pass<"detect-reduction"> {
  let summary = "Detect array reduction operations in a loop";
  let constructor = "mlir::polygeist::createDetectReductionPass()";
  let dependentDialects =
    ["affine::AffineDialect", "polygeist::PolygeistDialect",
     "scf::SCFDialect", "sycl::SYCLDialect"];
  let options = [
    Option<"relaxedAliasing", "relaxed-aliasing", "bool", 
            /*default=*/"false", 
            "Whether to use strict aliasing (i.e. type-based aliasing) rules or not">,
  ];
  let statistics = [
    Statistic<"numReductions", "num-reductions", "Number of reductions detected">
  ];
}

def KernelDisjointSpecialization : Pass<"kernel-disjoint-specialization", "mlir::ModuleOp"> {
  let summary = "Specialize SYCL kernel body functions";
  let description = [{
    Version the callsite from SYCL kernel to the SYCL kernel body function,
    where the specialized version ensures SYCL accessors do not overlap.
  }];
  let dependentDialects = 
      ["memref::MemRefDialect", "polygeist::PolygeistDialect",
       "scf::SCFDialect", "sycl::SYCLDialect"];
  let constructor = "mlir::polygeist::createKernelDisjointSpecializationPass()";
  let options = [
    Option<"relaxedAliasing", "relaxed-aliasing", "bool", 
            /*default=*/"false", 
            "Whether to use strict aliasing (i.e. type-based aliasing) rules or not">,
  ];
  let statistics = [
    Statistic<"numFunctionSpecialized", "num-function-specialized", "Number of function specialized">
  ];
}

def Mem2Reg : Pass<"polygeist-mem2reg"> {
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

def LoopInternalization : Pass<"loop-internalization"> {
  let summary = "Promote SYCL memory accesses in a loop nest to shared local memory";
  let description = [{
    SYCL memory accesses in perfectly nested loops are promoted to shared local
    memory by leveraging the loop tiling infrastructure.
  }];
  let constructor = "mlir::polygeist::createLoopInternalizationPass()";
  let options = [
    Option<"relaxedAliasing", "relaxed-aliasing", "bool", 
           /*default=*/"false", 
           "Whether to use strict aliasing (i.e. type-based aliasing) rules or not">,
    Option<"sharedMemorySize", "shared-memory-size", "unsigned",
           /*default=*/"32000", "The size of shared local memory in bytes">,
    Option<"unrollFactor", "unroll-factor", "unsigned",
           /*default=*/"4", "The number of times the tiled loop is unrolled">
  ];
  let statistics = [
    Statistic<"numAccessInternalized", "num-access-internalized", "Number of accesses internalized">,
    Statistic<"numLoopInternalized", "num-loop-internalized", "Number of loops internalized">
  ];
  let dependentDialects =
      ["affine::AffineDialect", "memref::MemRefDialect", "scf::SCFDialect", 
       "spirv::SPIRVDialect"];
}

def SCFBarrierRemovalContinuation : InterfacePass<"barrier-removal-continuation", "FunctionOpInterface"> {
  let summary = "Remove scf.barrier using continuations";
  let constructor = "mlir::polygeist::createBarrierRemovalContinuation()";
  let dependentDialects = ["memref::MemRefDialect"];
}

def SCFRaiseToAffine : Pass<"raise-scf-to-affine"> {
  let summary = "Raise SCF to affine";
  let constructor = "mlir::polygeist::createRaiseSCFToAffinePass()";
  let dependentDialects = ["affine::AffineDialect"];
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
  let dependentDialects =
    ["affine::AffineDialect", "polygeist::PolygeistDialect",
     "scf::SCFDialect", "sycl::SYCLDialect"];
  let constructor = "mlir::polygeist::createLICMPass()";
  let options = [
    Option<"relaxedAliasing", "relaxed-aliasing", "bool", 
            /*default=*/"false", 
            "Whether to use strict aliasing (i.e. type-based aliasing) rules or not">,
  ];
  let statistics = [
    Statistic<"numOpHoisted", "num-operations-hoisted", "Number of operations hoisted">
  ];
}

def SYCLRaiseHostConstructs : Pass<"sycl-raise-host"> {
  let summary = "Raise relevant sequences of host code to SYCL dialect operations";
  // TODO: Description
  let dependentDialects = ["arith::ArithDialect", "sycl::SYCLDialect", "vector::VectorDialect"];
  let constructor = "mlir::polygeist::createSYCLHostRaisingPass()";
  let options = RewritePassUtils.options;
  let statistics = [
    Statistic<"NumRaisedSetCapturedOps", "num-raised-set-captured-ops",
              "Number of raised `sycl.host.set_captured` ops">,
    Statistic<"NumRaisedSetNDRangeOps", "num-raised-set-nd-range-ops",
              "Number of raised `sycl.host.handler.set_nd_range` ops">,
    Statistic<"NumRaisedSetKernelOps", "num-raised-set-kernel-ops",
              "Number of raised `sycl.host.handler.set_kernel` ops">,
    Statistic<"NumRaisedScheduleKernelOps", "num-raised-schedule-kernel-ops",
              "Number of raised `sycl.host.schedule_kernel` ops">
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
