//===---------------- SYCLVirtualFunctionsAnalysis.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the SYCLVirtualFunctionsAnalysis
// pass that is responsible for checking that virtual functions are used
// properly in SYCL device code:
// - if a kernel submitted without the calls_indirectly property performs
//   virtual function calls, a diagnostic should be emitted.
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/SYCLVirtualFunctionsAnalysis.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"

using namespace llvm;

namespace {

using CallGraphTy = DenseMap<const Function *, SmallPtrSet<Value *, 8>>;

void emitDiagnostic(const SmallVector<const Function *> &Stack) {
  diagnoseSYCLIllegalVirtualFunctionCall(Stack);
}

void checkKernelImpl(const Function *F, const CallGraphTy &CG,
                     SmallVector<const Function *> &Stack) {
  Stack.push_back(F);
  CallGraphTy::const_iterator It = CG.find(F);
  // It could be that the function itself is a leaf and doesn't call anything
  if (It != CG.end()) {
    const SmallPtrSet<Value *, 8> &Callees = It->getSecond();
    for (const Value *V : Callees) {
      auto *Callee = dyn_cast<Function>(V);
      if (Callee)
        checkKernelImpl(Callee, CG, Stack);
      else
        emitDiagnostic(Stack);
    }
  }

  Stack.pop_back();
}

void checkKernel(const Function *F, const CallGraphTy &CG) {
  SmallVector<const Function *> CallStack;
  checkKernelImpl(F, CG, CallStack);
}

} // namespace

PreservedAnalyses
SYCLVirtualFunctionsAnalysisPass::run(Module &M, ModuleAnalysisManager &MAM) {
  CallGraphTy CallGraph;
  SmallVector<const Function *> Kernels;
  SetVector<const Function *> WorkList;

  // Identify list of kernels that we need to check
  for (const Function &F : M) {
    // We only traverse call graphs of SYCL kernels
    if (F.getCallingConv() != CallingConv::SPIR_KERNEL)
      continue;

    // If a kernel is annotated to use virtual functions, we skip it
    if (F.hasFnAttribute("calls-indirectly"))
      continue;

    // Otherwise, we build call graph for a kernel to ensure that it does not
    // perform virtual function calls since that is prohibited by the core
    // SYCL 2020 specification
    WorkList.insert(&F);
    Kernels.push_back(&F);
  }

  // Build call graph for each of them
  for (size_t I = 0; I < WorkList.size(); ++I) {
    const Function *F = WorkList[I];
    for (const Instruction &I : instructions(F)) {
      const auto *CI = dyn_cast<CallInst>(&I);
      if (!CI)
        continue;

      bool ToAdd = false;
      if (const auto *CF = CI->getCalledFunction()) {
        if (CF->isDeclaration())
          continue;

        WorkList.insert(CF);
        ToAdd = true;
      } else if (CI->isIndirectCall() && CI->hasFnAttr("virtual-call")) {
        ToAdd = true;
      }

      if (ToAdd)
        CallGraph[F].insert(CI->getCalledOperand());
    }
  }

  // Emit a diagnostic if a kernel performs virtual function calls
  for (const auto *K : Kernels) {
    checkKernel(K, CallGraph);
  }

  return PreservedAnalyses::all();
}
