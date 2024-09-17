//==-------------------------- ModuleHelper.cpp ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ModuleHelper.h"

#include "target/TargetFusionInfo.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/IR/Function.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

std::unique_ptr<Module>
helper::ModuleHelper::cloneAndPruneModule(Module *Mod,
                                          ArrayRef<Function *> CGRoots) {
  // Identify unused functions, i.e., functions not reachable from
  // one of the root nodes in CGRoots.
  SmallPtrSet<Function *, 16> UnusedFunctions;
  identifyUnusedFunctions(Mod, CGRoots, UnusedFunctions);

  {
    TargetFusionInfo TFI{Mod};
    SmallVector<Function *> Unused{UnusedFunctions.begin(),
                                   UnusedFunctions.end()};
    TFI.notifyFunctionsDelete(Unused);
  }

  // Clone the module, but use an external reference in place of the global
  // definition for unused functions.
  auto FunctionCloneMask = [&](const GlobalValue *GV) -> bool {
    if (const auto *F = dyn_cast<Function>(GV)) {
      return !UnusedFunctions.count(F);
    }
    return true;
  };
  ValueToValueMapTy VMap;
  auto NewMod = llvm::CloneModule(*Mod, VMap, FunctionCloneMask);

  // Remove the external references for unused functions from the clone module.
  for (auto *UF : UnusedFunctions) {
    auto *ExternalDef = NewMod->getFunction(UF->getName());
    assert(ExternalDef && "No external definition with this name");
    ExternalDef->eraseFromParent();
  }
  return NewMod;
}

void helper::ModuleHelper::identifyUnusedFunctions(
    Module *Mod, ArrayRef<Function *> CGRoots,
    SmallPtrSetImpl<llvm::Function *> &UnusedFunctions) {

  // Get the call-graph for this module.
  CallGraph CG(*Mod);
  // CG.dump();
  SmallPtrSet<Function *, 32> UsedFunctions;
  // Worklist algorithm: Retrieve a function from the work list, mark it as used
  // and add all functions called from this function to the worklist. Repeat
  // until the worklist is empty.
  SmallVector<Function *> Worklist;
  for (llvm::Function *Root : CGRoots) {
    Worklist.push_back(Root);
  }
  for (size_t I = 0; I < Worklist.size(); ++I) {
    llvm::Function *F = Worklist[I];
    UsedFunctions.insert(F);
    for (auto &CGR : *CG[F]) {
      auto *CF = CGR.second->getFunction();
      if (CF && !UsedFunctions.count(CF)) {
        // Only added previously unused functions to the worklist.
        Worklist.push_back(CF);
      }
    }
  }

  // Identify all functions in the input module, which have not been marked as
  // "used" in the previous step, as unused functions.
  // NOTE: LLVM intrinsic functions do not participate in the call-graph and
  // could falsely be detected as unused. Therefore, we never add them to the
  // list of unused functions and never erase them.
  for (auto &F : *Mod) {
    if (!UsedFunctions.count(&F) && !F.isIntrinsic()) {
      UnusedFunctions.insert(&F);
    }
  }
}
