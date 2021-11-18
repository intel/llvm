//===---- DelimitESIMDandSYCL.cpp - delimit ESIMD and SYCL code -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Implements the ESIMD/SYCL delimitor pass. See pass description in the header.
//===----------------------------------------------------------------------===//

#include "llvm/SyclLowerIR/DelimitEsimdandSycl.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/GenXIntrinsics/GenXMetadata.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <iostream>

#define DEBUG_TYPE "delimit-esimd-and-sycl"

using namespace llvm;

namespace {
SmallPtrSet<Type *, 4> collectGenXVolatileTypes(Module &);
void generateKernelMetadata(Module &);

class DelimitESIMDandSYCLLegacyPass : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  DelimitESIMDandSYCLLegacyPass() : ModulePass(ID) {
    initializeDelimitESIMDandSYCLLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  // run the DelimitESIMDandSYCL pass on the specified module
  bool runOnModule(Module &M) override {
    ModuleAnalysisManager MAM;
    auto PA = Impl.run(M, MAM);
    return !PA.areAllPreserved();
  }

private:
  DelimitESIMDandSYCLPass Impl;
};
} // namespace

char DelimitESIMDandSYCLLegacyPass::ID = 0;
INITIALIZE_PASS(DelimitESIMDandSYCLLegacyPass, "DelimitESIMDandSYCL",
                "Delimit ESIMD and SYCL code in a module", false, false)

// Public interface to the DelimitESIMDandSYCLPass.
ModulePass *llvm::createDelimitESIMDandSYCLPass() {
  return new DelimitESIMDandSYCLLegacyPass();
}

namespace {

constexpr char ESIMD_MARKER_MD[] = "sycl_explicit_simd";
constexpr char INVOKE_SIMD_PREF[] = "_Z21__builtin_invoke_simd";

using FuncPtrSet = SmallPtrSetImpl<Function*>;

template <class ActOnCallF>
void traverseCalls(Function *F, ActOnCallF Action) {
  for (const auto &I : instructions(F)) {
    if (const CallBase *CB = dyn_cast<CallBase>(&I)) {
      if (Function *CF = CB->getCalledFunction()) {
        if (!CF->isDeclaration())
          Action(CB, CF);
      } else {
        llvm_unreachable("Unsupported call form");
      }
    }
  }
}

void prnfset(const char *msg, FuncPtrSet &S) {
  std::cout << msg << ":\n";
  for (const auto *F : S) {
    std::cout << "  " << F->getName().data() << "\n";
  }
}

bool isCast(const Value *V) {
  int Opc = Operator::getOpcode(V);
  return (Opc == Instruction::BitCast) || (Opc == Instruction::AddrSpaceCast);
}

using ValueSetImpl = SmallPtrSetImpl<Value*>;
using ValueSet = SmallPtrSet<Value*, 4>;
using ConstValueSetImpl = SmallPtrSetImpl<const Value*>;
using ConstValueSet = SmallPtrSet<const Value*, 4>;

Value* stripCasts(Value* V) {
  if (!V->getType()->isPtrOrPtrVectorTy())
    return V;
  // Even though we don't look through PHI nodes, we could be called on an
  // instruction in an unreachable block, which may be on a cycle.
  ConstValueSet Visited;
  Visited.insert(V);

  do {
    if (isCast(V)) {
      V = cast<Operator>(V)->getOperand(0);
    }
    assert(V->getType()->isPtrOrPtrVectorTy() && "Unexpected operand type!");
  } while (Visited.insert(V).second);
  return V;
}

const Value* getSingleUserSkipCasts(const Value *V) {
  while (isCast(V)) {
    if (V->getNumUses() != 1) {
      return nullptr;
    }
    V = *(V->user_begin());
  }
  return V;
}

using UseSet = SmallPtrSet<Use*, 4>;
using UseSetImpl = SmallPtrSetImpl<const Use*>;

void collectUsesSkipThroughCasts(Value *V, UseSet &Uses) {
  for (Use &U : V->uses()) {
    Value *VV = U.getUser();

    if (isCast(VV)) {
      collectUsesSkipThroughCasts(VV, Uses);
    } else {
      Uses.insert(&U);
    }
  }
}

Value* getInvokeeIfInvokeSimdCall(const CallInst *CI) {
  Function *F = CI->getCalledFunction();

  if (F && F->getName().startswith(INVOKE_SIMD_PREF)) {
    return CI->getArgOperand(0);
  }
  return nullptr;
}

void getPossibleStoredVals(Value *Addr, ValueSetImpl &Vals) {
  ValueSet Visited;
  AllocaInst *LocalVar = dyn_cast_or_null<AllocaInst>(stripCasts(Addr));

  if (!LocalVar) {
    llvm_unreachable("unsupported data flow pattern for invoke_simd 10");
  }
  UseSet Uses;
  collectUsesSkipThroughCasts(LocalVar, Uses);

  for (Use *U : Uses) {
    Value *V = U->getUser();

    if (auto *StI = dyn_cast<StoreInst>(V)) {
      constexpr int StoreInstValueOperandIndex = 0;

      if (U != &StI->getOperandUse(StoreInst::getPointerOperandIndex())) {
        assert(U == &StI->getOperandUse(StoreInstValueOperandIndex));
        // this is double indirection - not supported
        llvm_unreachable("unsupported data flow pattern for invoke_simd 11");
      }
      V = stripCasts(StI->getValueOperand());

      if (auto *LI = dyn_cast<LoadInst>(V)) {
        // A value loaded from another address is stored at this address - recurse
        // into the other addresss
        getPossibleStoredVals(LI->getPointerOperand(), Vals);
      } else {
        Vals.insert(V);
      }
      continue;
    }
    if (const auto *CI = dyn_cast<CallInst>(V)) {
      // only __builtin_invoke_simd is allowed, otherwise the pointer escapes
      if (!getInvokeeIfInvokeSimdCall(CI)) {
        llvm_unreachable("unsupported data flow pattern for invoke_simd 12");
      }
      continue;
    }
    if (const auto *LI = dyn_cast<LoadInst>(V)) {
      // LoadInst from this addr is OK, as it does not affect what can be stored
      // through the addr
      continue;
    }
    llvm_unreachable("unsupported data flow pattern for invoke_simd 13");
  }
}

// Example1 (function is direct argument to _Z21__builtin_invoke_simd):
// %call6.i = call spir_func float @_Z21__builtin_invoke_simd...(
//   <16 x float> (float addrspace(4)*, <16 x float>, i32)* %28, <== function pointer
//   float addrspace(4)* %arg1,
//   float %arg2,
//   i32 %arg3)
//
// Example 2 (invoke_simd's target function pointer flows through IR):
// %"fptr_t = <16 x float> (float addrspace(4)*, <16 x float>, i32)*
// ...
// %fa_as0 = alloca %fptr_t
// ...
// %fa = addrspacecast %fptr_t* %fa_as0 to %fptr_t addrspace(4)*
// ...
// store %fptr_t @__SIMD_CALLEE, %fptr_t addrspace(4)* %fa
// ...
// %f = load %fptr_t, %fptr_t addrspace(4)* %fa
// ...
// %res = call spir_func float @_Z21__builtin_invoke_simd...(
//   %fptr_t %f, <== function pointer
//  float addrspace(4)* %arg1,
//  float %arg2,
//  i32 %arg3)
//
void processInvokeSimdCall(CallInst *CI) {
  Value *V = getInvokeeIfInvokeSimdCall(CI);

  if (!V) {
    llvm_unreachable(("bad use of " + Twine(INVOKE_SIMD_PREF)).str().c_str());
  }
  Function *SimdF = nullptr;

  if (!(SimdF = dyn_cast<Function>(V))) {
    LoadInst *LI = dyn_cast<LoadInst>(stripCasts(V));

    if (!LI) {
      llvm_unreachable("unsupported data flow pattern for invoke_simd 0");
    }
    ValueSet Vals;
    getPossibleStoredVals(LI->getPointerOperand(), Vals);

    if (Vals.size() != 1 || !(SimdF = dyn_cast<Function>(*Vals.begin()))) {
      llvm_unreachable("unsupported data flow pattern for invoke_simd 1");
    }
    // _Z21__builtin_invoke_simd invokee is an SSA value, replace it with the
    // link-time constant SimdF as computed by getPossibleStoredVals
    CallInst *CI1 = cast<CallInst>(CI->clone());
    constexpr int SimdInvokeInvokeeArgIndex = 0;
    CI1->setOperand(SimdInvokeInvokeeArgIndex, SimdF);
    CI1->insertAfter(CI);
    CI->replaceAllUsesWith(CI1);
    CI->eraseFromParent();
  }
  if (!SimdF->hasFnAttribute(llvm::genx::VCFunctionMD::VCStackCall)) {
    SimdF->addFnAttr(llvm::genx::VCFunctionMD::VCStackCall);
  }
}

// Given module M and function IsRootInA, let's define:
// M - a set of all functions in the module
// A-roots - a set of functions from M, for which IsRootInA returns true
//
// This function divides M based on "calls to" relation (call graph) into 3
// parts (sets):
// B - all functions which are neither A-roots nor reachable from any A-root
// AB - functions reachable both from B and from A-roots
// A - A-roots plus all functions recheable from them excuding those also
//     reachable from B
// The following holds true for the result:
// - A + B + AB = M
// - A, B and AB are disjoint
// - there is no path in the callgraph from A to B or back
// - for every function F from AB, there is at least one path from A to F and
//   at least one path from B to F
//
template <class CfgARootTestF>
void divideModuleCallGraph(Module &M, FuncPtrSet &A, FuncPtrSet &AB, SmallPtrSet<Function*, 32> &B, CfgARootTestF IsRootInA) {
  {
    SmallVector<Function*, 32> Workq;

    // Collect CFG roots and populate work queue with them.
    for (Function &F : M) {
      if (F.isDeclaration()) {
        // TODO processing of invoke_simd should be moved into a separate pass generic
        // for all BEs (maybe with some custom parts)
        if (F.getName().startswith(INVOKE_SIMD_PREF)) {
          for (User *Usr : F.users()) {
            // a call can be the only use of invoke_simd built-in
            CallInst *CI = cast<CallInst>(Usr);
            processInvokeSimdCall(CI);
          }
        }
        continue;
      }
      if (IsRootInA(&F)) {
        A.insert(&F);
        Workq.push_back(&F);
      } else {
        B.insert(&F); // all non-A-roots go to B for now, clean up below
      }
    }
    // Build and traverse the CFGs.
    while (Workq.size() > 0) {
      Function *F = Workq.pop_back_val();
      B.erase(F); // cleanup B: F is reached from A, then it can't be part of B
      traverseCalls(F, [&A, &Workq](const CallBase *CB, Function *F1) {
        if (A.count(F1) == 0) {
          A.insert(F1);
          Workq.push_back(F1);
        }
      });
    }
  }
  // B is ready at this point, but some of A functions can be also reacheable
  // from B (A is now actually A' = A + AB) - identify them, remove from A and
  // add to AB.
  {
    SmallVector<Function*, 32> Workq(B.begin(), B.end());

    while (Workq.size() > 0) {
      Function *F = Workq.pop_back_val();

      traverseCalls(F, [&A, &B, &AB, &Workq](const CallBase *CB, Function *F1) {
        if (B.count(F1) == 0 && AB.count(F1) == 0) {
          // F1 is reachable from B (by Workq construction), but not part of B
          // and hasn't been met yet - must be part of A'
          if (!A.erase(F1))
            llvm_unreachable("callgraph division algorithm error");
          // F1 was part of A - it is reacheable both from A and B
          AB.insert(F1);
          Workq.push_back(F1);
        }
        // else F1 is either part of B or already met - not adding to Workq
      });
    }
  }
  // prnfset("========== A:", A);
  // prnfset("\n========== B:", B);
  // prnfset("\n========== AB:", AB);
}

Function* clone(Function *F, Twine suff) {
  ValueToValueMapTy VMap;
  Function *Res = CloneFunction(F, VMap);
  Res->setName(F->getName() + "." + suff);
  Res->copyAttributesFrom(F);
  Res->setLinkage(F->getLinkage());
  Res->setVisibility(F->getVisibility());
  return Res;
}
} // namespace

PreservedAnalyses DelimitESIMDandSYCLPass::run(Module &M, ModuleAnalysisManager &) {
  SmallPtrSet<Function*, 32> EsimdOnlyFuncs; // called only from ESIMD callgraph
  SmallPtrSet<Function*, 32> SyclOnlyFuncs; // called only from SYCL callgraph
  SmallPtrSet<Function*, 32> CommonFuncs; // called both from ESIMD and SYCL

  // Collect the 3 sets of functions based on CFG: 
  divideModuleCallGraph(M, EsimdOnlyFuncs, CommonFuncs, SyclOnlyFuncs, [](Function *F) {
    return F->getMetadata(ESIMD_MARKER_MD) != nullptr;
  });
  if (EsimdOnlyFuncs.size() == 0)
    return PreservedAnalyses::all();

  DenseMap<Value*, Value*> Sycl2Esimd; // map common function to its Esimd clone

  // Clone common functions:
  for (auto *F : CommonFuncs) {
    Function *EsimdF = clone(F, "esimd");
    EsimdOnlyFuncs.erase(F);
    EsimdOnlyFuncs.insert(EsimdF);
    Sycl2Esimd[F] = EsimdF;
  }
  bool Modified = false;

  // Mark all Esimd functions with proper attribute - VCFunction:
  for (auto *F : EsimdOnlyFuncs) {
    F->addFnAttr(llvm::genx::VCFunctionMD::VCFunction);
    Modified = true;
  }
  // Now replace common functions usages within the Esimd call graph with the
  // clones.
  // TODO now the "usage" means only calls, function pointers are not supported.
  for (auto *F : CommonFuncs) {
    auto *EsimdF = Sycl2Esimd[F];
    F->replaceUsesWithIf(EsimdF, [F, &Modified, &EsimdOnlyFuncs](Use &U) -> bool {
      if (const CallBase *CB = dyn_cast<const CallBase>(U.getUser())) {
        Function *CF = CB->getCalledFunction();
        if (CF != F)
          llvm_unreachable("Unsupported call form");
        // see if the call happens within a function from the ESIMD call graph:
        bool CalledFromEsimd = EsimdOnlyFuncs.count(CB->getFunction()) > 0;
        Modified |= CalledFromEsimd;
        return CalledFromEsimd;
      }
      llvm_unreachable("Unsupported use of function");
      return false;
    });
  }
  return Modified ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
