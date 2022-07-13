//===------------ ESIMDUtils.cpp - ESIMD utility functions ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Utility functions for processing ESIMD code.
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/ESIMD/ESIMDUtils.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"

namespace llvm {
namespace esimd {

void traverseCallgraphUp(llvm::Function *F, CallGraphNodeAction ActionF,
                         bool ErrorOnNonCallUse) {
  SmallPtrSet<Function *, 32> FunctionsVisited;
  SmallVector<Function *, 32> Worklist{F};

  while (!Worklist.empty()) {
    Function *CurF = Worklist.pop_back_val();
    FunctionsVisited.insert(CurF);
    // Apply the action function.
    ActionF(CurF);

    // Update all callers as well.
    for (auto It = CurF->use_begin(); It != CurF->use_end(); It++) {
      auto FCall = It->getUser();
      auto ErrMsg =
          llvm::Twine(__FILE__ " ") +
          "Function use other than call detected while traversing call\n"
          "graph up to a kernel";
      if (!isa<CallInst>(FCall)) {
        // A use other than a call is met...
        if (ErrorOnNonCallUse) {
          // ... non-call is an error - report
          llvm::report_fatal_error(ErrMsg);
        } else {
          // ... non-call is OK - add using function to the worklist
          if (auto *I = dyn_cast<Instruction>(FCall)) {
            auto UseF = I->getFunction();

            if (!FunctionsVisited.count(UseF)) {
              Worklist.push_back(UseF);
            }
          }
        }
      } else {
        auto *CI = cast<CallInst>(FCall);

        if ((CI->getCalledFunction() != CurF) && ErrorOnNonCallUse) {
          // CurF is used in a call, but not as the callee.
          llvm::report_fatal_error(ErrMsg);
        } else {
          auto FCaller = CI->getFunction();

          if (!FunctionsVisited.count(FCaller)) {
            Worklist.push_back(FCaller);
          }
        }
      }
    }
  }
}

bool isESIMDKernel(const Function &F) {
  return (F.getCallingConv() == CallingConv::SPIR_KERNEL) &&
         (F.getMetadata("sycl_explicit_simd") != nullptr);
}

} // namespace esimd
} // namespace llvm
