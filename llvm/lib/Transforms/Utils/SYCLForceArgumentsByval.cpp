//=== SYCLForceArgumentsByval.cpp - forces kernel arguments to be by value ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Simple pass to enforce by-value passing of arguments in SYCL kernels.
// If a pointer argument of a kernel does not have the byval attribute and it in
// addrspace(0) it will be given a new byval attribute with a type corresponding
// to its elements.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/SYCLForceArgumentsByval.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "sycl-force-args-byval"

namespace {
class SYCLForceArgumentsByvalLegacyPass : public FunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid
  SYCLForceArgumentsByvalLegacyPass() : FunctionPass(ID) {
    initializeSYCLForceArgumentsByvalLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  // run the LowerWGScope pass on the specified module
  bool runOnFunction(Function &F) override {
    FunctionAnalysisManager FAM;
    auto PA = Impl.run(F, FAM);
    return !PA.areAllPreserved();
  }

private:
  SYCLForceArgumentsByvalPass Impl;
};
} // namespace

char SYCLForceArgumentsByvalLegacyPass::ID = 0;
INITIALIZE_PASS(SYCLForceArgumentsByvalLegacyPass, "SYCLForceArgumentsByval",
                "Force SYCL Kernel Arguments By Value", false, false)

// Public interface to the SYCLForceArgumentsByvalPass.
FunctionPass *llvm::createSYCLForceArgumentsByvalPass() {
  return new SYCLForceArgumentsByvalLegacyPass();
}

PreservedAnalyses
SYCLForceArgumentsByvalPass::run(Function &F, FunctionAnalysisManager &FAM) {

  // Byval arguments should only be forced on kernel arguments
  if (F.getCallingConv() != CallingConv::SPIR_KERNEL) {
    return PreservedAnalyses::all();
  }

  bool Changed = false;

  for (Argument &Arg : F.args()) {
    // Ignore arguments that are already by-value
    if (Arg.hasByValAttr())
      continue;

    // Ignore non-pointer types and pointers with address spaces other than
    // addrspace(0). Pointers with other address spaces are assumed to be SYCL
    // accessors.
    auto ArgT = Arg.getType();
    if (!ArgT->isPointerTy() || cast<PointerType>(ArgT)->getAddressSpace())
      continue;

    Arg.addAttrs(
        llvm::AttrBuilder().addByValAttr(ArgT->getPointerElementType()));
    Changed = true;
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
