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

using FuncPtrSet = SmallPtrSetImpl<Function*>;

template <class ActOnCallF>
void traverseCalls(Function *F, ActOnCallF Action) {
  for (const auto &I : instructions(F)) {
    if (const CallBase *CB = dyn_cast<CallBase>(&I)) {
      if (Function *CF = CB->getCalledFunction()) {
        if (!CF->isDeclaration())
          Action(CF);
      } else {
        llvm_unreachable_internal("Unsupported call form", __FILE__, __LINE__);
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
      if (F.isDeclaration())
        continue;
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
      traverseCalls(F, [&A, &Workq](Function *F1) {
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

      traverseCalls(F, [&A, &B, &AB, &Workq](Function *F1) {
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

  // Mark all Esimd functions with proper attribute:
  for (auto *F : EsimdOnlyFuncs) {
    F->addFnAttr(ESIMD_MARKER_MD);
    Modified = true;
  }
  // TODO update GenXSPIRVWriterAdaptor and IGC SYCL/ESIMD splitter to use
  // sycl_explicit_simd (or some other attribute) to separate ESIMD functions
  // from SYCL so that there is no need/ to add CMGenxSIMT attribute to every
  // non-ESIMD function. llvm::genx::VCFunctionMD::VCFunction does not work,
  // because GenXSPIRVWriterAdaptor adds it to all functions.
  for (auto *F : SyclOnlyFuncs) {
    F->addFnAttr(llvm::genx::FunctionMD::CMGenxSIMT);
    Modified = true;
  }
  // Now replace common functions usages within Esimd CFGs with the clones.
  // TODO now the "usage" means only calls, function pointers are not supported.
  for (auto *F : CommonFuncs) {
    auto *EsimdF = Sycl2Esimd[F];
    F->replaceUsesWithIf(EsimdF, [F, &Modified, &EsimdOnlyFuncs](Use &U) -> bool {
      if (const CallBase *CB = dyn_cast<const CallBase>(U.getUser())) {
        Function *CF = CB->getCalledFunction();
        if (CF != F)
          llvm_unreachable_internal("Unsupported call form", __FILE__, __LINE__);
        // see if the call happens within a function from the ESIMD call graph:
        bool CalledFromEsimd = EsimdOnlyFuncs.count(CB->getFunction()) > 0;
        Modified |= CalledFromEsimd;
        return CalledFromEsimd;
      }
      llvm_unreachable_internal("Unsupported use of function", __FILE__, __LINE__);
      return false;
    });
    F->addFnAttr(llvm::genx::FunctionMD::CMGenxSIMT);
    Modified = true; // see TODO above
  }
  return Modified ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
