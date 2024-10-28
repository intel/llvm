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
// Additionally, the pass sets "calls-indirectly" attribute for kernels which
// create, but don't call virtual functions. This attribute is needed to emit
// the right device image properties later which will be crucial to ensure
// proper runtime linking.
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/SYCLVirtualFunctionsAnalysis.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Pass.h"

using namespace llvm;

namespace {

using CallGraphTy = DenseMap<const Function *, SmallPtrSet<Value *, 8>>;
using FuncToFuncMapTy = DenseMap<const Function *, SmallPtrSet<Function *, 8>>;

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

void computeFunctionToKernelsMappingImpl(Function *Kernel, const Function *F,
                                         const CallGraphTy &CG,
                                         FuncToFuncMapTy &Mapping) {
  Mapping[F].insert(Kernel);
  CallGraphTy::const_iterator It = CG.find(F);
  // It could be that the function itself is a leaf and doesn't call anything
  if (It == CG.end())
    return;

  const SmallPtrSet<Value *, 8> &Callees = It->getSecond();
  for (const Value *V : Callees)
    if (auto *Callee = dyn_cast<Function>(V))
      computeFunctionToKernelsMappingImpl(Kernel, Callee, CG, Mapping);
}

// Compute a map from functions used by a kernel to that kernel.
// For simplicity we also consider a kernel to be using itself.
void computeFunctionToKernelsMapping(Function *Kernel, const CallGraphTy &CG,
                                     FuncToFuncMapTy &Mapping) {
  computeFunctionToKernelsMappingImpl(Kernel, Kernel, CG, Mapping);
}

void collectVTablesThatUseFunction(
    const Value *V, SmallVectorImpl<const GlobalVariable *> &VTables) {
  for (const auto *U : V->users()) {
    // GlobalVariable is also a constant
    if (const auto *GV = dyn_cast<GlobalVariable>(U)) {
      // The core SYCL specification prohibits ODR use of non-const global
      // variables in SYCL kernels. There are extensions like device_global that
      // lift some of the limitations, but we still assume that there are no
      // globals that reference function pointers other than virtual tables.
      VTables.push_back(GV);
    } else if (isa<ConstantExpr>(U)) {
      // Constant expression like
      // ptr addrspace(4) addrspacecast (ptr @foo to ptr addrspace(4))
      // Could be a part of vtable initializer
      collectVTablesThatUseFunction(U, VTables);
    } else if (isa<Constant>(U)) {
      // [3 x ptr addrspace(4)] [
      //    ptr addrspace(4) addrspacecast (ptr @foo to ptr addrspace(4)), ...]
      collectVTablesThatUseFunction(U, VTables);
    } else {
      llvm_unreachable("Unhandled type of user");
    }
  }
}

// The same ConstantExpr could be used by two functions
void collectEnclosingFunctions(const Value *V,
                               SmallPtrSetImpl<const Function *> &Functions) {
  if (isa<ConstantExpr>(V)) {
    for (const auto *U : V->users())
      collectEnclosingFunctions(U, Functions);
    return;
  }

  if (auto *I = dyn_cast<Instruction>(V)) {
    Functions.insert(I->getFunction());
    return;
  }

  llvm_unreachable("Unhandled type of value");
}

} // namespace

PreservedAnalyses
SYCLVirtualFunctionsAnalysisPass::run(Module &M, ModuleAnalysisManager &MAM) {
  CallGraphTy CallGraph;
  SmallVector<Function *> AllKernels;
  SmallVector<Function *> KernelsToCheck;
  SmallVector<const Function *> IndirectlyCallableFuncs;
  SetVector<const Function *> WorkList;

  // Identify list of kernels that we need to check
  for (Function &F : M) {
    if (F.hasFnAttribute("indirectly-callable"))
      IndirectlyCallableFuncs.push_back(&F);

    // We only traverse call graphs of SYCL kernels
    if (F.getCallingConv() != CallingConv::SPIR_KERNEL)
      continue;

    // We record all the kernels here, because we may end up propagating
    // calls-indirectly to them if they use vtables.
    AllKernels.push_back(&F);
    WorkList.insert(&F);

    // However, we only need to check kernel's call graph if it is not annotated
    // to use virtual functions to ensure that it indeed doesn't use them.
    if (!F.hasFnAttribute("calls-indirectly"))
      KernelsToCheck.push_back(&F);
  }

  // If there are no virtual functions to call in a module, then we can skip
  // the whole analysis
  if (IndirectlyCallableFuncs.empty())
    return PreservedAnalyses::all();

  // Build call graph for each kernel
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
  for (auto *K : KernelsToCheck)
    checkKernel(K, CallGraph);

  // Cache to know which function is used by which kernels
  FuncToFuncMapTy FunctionToKernels;
  for (auto *K : AllKernels)
    computeFunctionToKernelsMapping(K, CallGraph, FunctionToKernels);

  for (const auto *F : IndirectlyCallableFuncs) {
    StringRef Set = F->getFnAttribute("indirectly-callable").getValueAsString();

    SmallVector<const GlobalVariable *, 4> VTables;
    collectVTablesThatUseFunction(F, VTables);
    SmallPtrSet<Function *, 8> KernelsToUpdate;

    for (const auto *GV : VTables) {
      // Find functions that use those vtables
      SmallPtrSet<const Function *, 16> FunctionsThatUseVTables;
      for (const auto *UU : GV->users())
        collectEnclosingFunctions(UU, FunctionsThatUseVTables);
      // And collect kernels that use those functions
      for (const Function *FF : FunctionsThatUseVTables)
        for (auto *K : FunctionToKernels[FF])
          KernelsToUpdate.insert(K);
    }

    // Update or attach "calls-indirectly" attribute to those kernels
    // indicating that they use virtual functions set 'Set'
    for (Function *K : KernelsToUpdate) {
      if (!K->hasFnAttribute("calls-indirectly"))
        K->addFnAttr("calls-indirectly", Set);
      else {
        StringRef UsedSets =
            K->getFnAttribute("calls-indirectly").getValueAsString();
        if (UsedSets.contains(Set))
          continue;

        K->removeFnAttr("calls-indirectly");
        SmallString<64> NewAttr = UsedSets;
        NewAttr += ",";
        NewAttr += Set;
        K->addFnAttr("calls-indirectly", NewAttr.str());
      }
    }
  }

  return PreservedAnalyses::all();
}
