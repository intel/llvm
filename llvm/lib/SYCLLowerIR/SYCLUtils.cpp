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
bool isAddressArgumentInvokeSIMD(const CallInst *CI) {
  constexpr char INVOKE_SIMD_PREF[] = "_Z33__regcall3____builtin_invoke_simd";

  Function *F = CI->getCalledFunction();

  if (F && F->getName().startswith(INVOKE_SIMD_PREF)) {
    return true;
  }
  return false;
}

bool filterFunctionPointer(Value *address) {
  if (address == nullptr) {
    return true;
  }

  SmallPtrSet<const Use *, 4> Uses;
  collectUsesLookThroughCasts(address, Uses);

  for (const Use *U : Uses) {
    Value *V = U->getUser();

    if (auto *StI = dyn_cast<StoreInst>(V)) {
      if (U != &StI->getOperandUse(StoreInst::getPointerOperandIndex())) {
        // this is double indirection - not supported
        return false;
      }
      V = stripCasts(StI->getPointerOperand());
      if (!isa<AllocaInst>(V)) {
        return false; // unsupported case of data flow through non-local memory
      }

      if (auto *LI = dyn_cast<LoadInst>(V)) {
        // A value loaded from another address is stored at this address -
        // recurse into the other address
        if (!filterFunctionPointer(LI->getPointerOperand())) {
          return false;
        }
      }
    } else if (const auto *CI = dyn_cast<CallInst>(V)) {
      // if __builtin_invoke_simd uses the pointer, do not traverse the function
      if (isAddressArgumentInvokeSIMD(CI)) {
        return false;
      }
    } else if (isa<LoadInst>(V)) {
      if (!filterFunctionPointer(V)) {
        return false;
      }
    } else {
      return false;
    }
  }

  return true;
}

void traverseCallgraphUp(llvm::Function *F, CallGraphNodeAction ActionF,
                         SmallPtrSetImpl<Function *> &FunctionsVisited,
                         bool ErrorOnNonCallUse) {
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
            if (auto *SI = dyn_cast<StoreInst>(I)) {
              Value *addr = SI->getPointerOperand();
              if (!filterFunctionPointer(addr)) {
                continue;
              }
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
