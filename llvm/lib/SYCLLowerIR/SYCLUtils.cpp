//===------------ SYCLUtils.cpp - SYCL utility functions ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Utility functions for SYCL.
//===----------------------------------------------------------------------===//
#include "llvm/SYCLLowerIR/SYCLUtils.h"
#include "llvm/IR/Instructions.h"
#include "llvm/SYCLLowerIR/ESIMD/ESIMDUtils.h"

namespace llvm {
namespace sycl {
namespace utils {

using namespace llvm::esimd;

void traverseCallgraphUp(llvm::Function *F, CallGraphNodeAction ActionF,
                         SmallPtrSetImpl<Function *> &FunctionsVisited,
                         bool ErrorOnNonCallUse,
                         const CallGraphFunctionFilter &functionFilter) {
  SmallVector<Function *, 32> Worklist;

  if (FunctionsVisited.count(F) == 0)
    Worklist.push_back(F);

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
            if (!functionFilter(I, CurF)) {
              continue;
            }

            auto UseF = I->getFunction();

            if (FunctionsVisited.count(UseF) == 0) {
              Worklist.push_back(UseF);
            }
          }
        }
      } else {
        auto *CI = cast<CallInst>(FCall);

        if ((CI->getCalledFunction() != CurF)) {
          // CurF is used in a call, but not as the callee.
          if (ErrorOnNonCallUse)
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
} // namespace utils
} // namespace sycl
} // namespace llvm
