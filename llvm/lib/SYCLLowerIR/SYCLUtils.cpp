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
#include "llvm/IR/Module.h"
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

bool isCast(const Value *V) {
  int Opc = Operator::getOpcode(V);
  return (Opc == Instruction::BitCast) || (Opc == Instruction::AddrSpaceCast);
}

bool isZeroGEP(const Value *V) {
  const auto *GEPI = dyn_cast<GetElementPtrInst>(V);
  return GEPI && GEPI->hasAllZeroIndices();
}

Value *stripCasts(Value *V) {
  return const_cast<Value *>(stripCasts(const_cast<const Value *>(V)));
}

const Value *stripCastsAndZeroGEPs(const Value *V);

Value *stripCastsAndZeroGEPs(Value *V) {
  return const_cast<Value *>(
      stripCastsAndZeroGEPs(const_cast<const Value *>(V)));
}

const Value *stripCasts(const Value *V) {
  if (!V->getType()->isPtrOrPtrVectorTy())
    return V;
  // Even though we don't look through PHI nodes, we could be called on an
  // instruction in an unreachable block, which may be on a cycle.
  SmallPtrSet<const Value *, 4> Visited;
  Visited.insert(V);

  do {
    if (isCast(V)) {
      V = cast<Operator>(V)->getOperand(0);
    }
    assert(V->getType()->isPtrOrPtrVectorTy() && "Unexpected operand type!");
  } while (Visited.insert(V).second);
  return V;
}

const Value *stripCastsAndZeroGEPs(const Value *V) {
  if (!V->getType()->isPtrOrPtrVectorTy())
    return V;
  // Even though we don't look through PHI nodes, we could be called on an
  // instruction in an unreachable block, which may be on a cycle.
  SmallPtrSet<const Value *, 4> Visited;
  Visited.insert(V);

  do {
    if (isCast(V)) {
      V = cast<Operator>(V)->getOperand(0);
    } else if (isZeroGEP(V)) {
      V = cast<GetElementPtrInst>(V)->getOperand(0);
    }
    assert(V->getType()->isPtrOrPtrVectorTy() && "Unexpected operand type!");
  } while (Visited.insert(V).second);
  return V;
}

void collectUsesLookThroughCasts(const Value *V,
                                 SmallPtrSetImpl<const Use *> &Uses) {
  for (const Use &U : V->uses()) {
    Value *VV = U.getUser();

    if (isCast(VV)) {
      collectUsesLookThroughCasts(VV, Uses);
    } else {
      Uses.insert(&U);
    }
  }
}

void collectUsesLookThroughCastsAndZeroGEPs(
    const Value *V, SmallPtrSetImpl<const Use *> &Uses) {
  assert(V->getType()->isPtrOrPtrVectorTy() && "pointer type expected");

  for (const Use &U : V->uses()) {
    Value *VV = U.getUser();

    if (isCast(VV) || isZeroGEP(VV)) {
      collectUsesLookThroughCastsAndZeroGEPs(VV, Uses);
    } else {
      Uses.insert(&U);
    }
  }
}

// Tries to find possible values stored into given address.
// Returns true if the set of values could be reliably found, false otherwise.
bool collectPossibleStoredVals(
    Value *Addr, SmallPtrSetImpl<Value *> &Vals,
    std::function<bool(const CallInst *)> EscapesIfAddrIsArgOf) {
  SmallPtrSet<Value *, 4> Visited;
  AllocaInst *LocalVar = dyn_cast_or_null<AllocaInst>(stripCasts(Addr));

  if (!LocalVar) {
    return false;
  }
  SmallPtrSet<const Use *, 4> Uses;
  collectUsesLookThroughCasts(LocalVar, Uses);

  for (const Use *U : Uses) {
    Value *V = U->getUser();

    if (auto *StI = dyn_cast<StoreInst>(V)) {
      constexpr int StoreInstValueOperandIndex = 0;

      if (U != &StI->getOperandUse(StoreInst::getPointerOperandIndex())) {
        assert(U == &StI->getOperandUse(StoreInstValueOperandIndex));
        // this is double indirection - not supported
        return false;
      }
      V = stripCasts(StI->getValueOperand());

      if (auto *LI = dyn_cast<LoadInst>(V)) {
        // A value loaded from another address is stored at this address -
        // recurse into the other address
        if (!collectPossibleStoredVals(LI->getPointerOperand(), Vals)) {
          return false;
        }
      } else {
        Vals.insert(V);
      }
      continue;
    }
    if (const auto *CI = dyn_cast<CallInst>(V)) {
      if (EscapesIfAddrIsArgOf(CI)) {
        return false;
      }
      continue;
    }
    if (isa<LoadInst>(V)) {
      // LoadInst from this addr is OK, as it does not affect what can be stored
      // through the addr
      continue;
    }
    return false;
  }
  return true;
}

} // namespace utils
} // namespace sycl
} // namespace llvm
