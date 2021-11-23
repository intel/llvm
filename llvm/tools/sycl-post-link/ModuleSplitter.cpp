//===---- ModuleSplitter.cpp - split a module into callgraphs -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Implements the ESIMD/SYCL delimitor pass. See pass description in the header.
//===----------------------------------------------------------------------===//

#include "ModuleSplitter.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/GenXIntrinsics/GenXMetadata.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"

#include <iostream>

#define DEBUG_TYPE "delimit-esimd-and-sycl"

using namespace llvm;
using namespace llvm::module_split;

namespace {

void cloneSubmoduleInto(const Module &M, const SetVector<const GlobalValue *> &SubmoduleGVs, ModuleDesc /*inout*/&Dst) {
  ValueToValueMapTy VMap;
  // Clone definitions only for needed globals. Others will be added as
  // declarations and removed later.
  std::unique_ptr<Module> MClone = CloneModule(
    M, VMap, [&](const GlobalValue *GV) {
    return SubmoduleGVs.count(GV);
  });
  // TODO: Use the new PassManager instead?
  legacy::PassManager Passes;
  // Do cleanup.
  Passes.add(createGlobalDCEPass());           // Delete unreachable globals.
  Passes.add(createStripDeadDebugInfoPass());  // Remove dead debug info.
  Passes.add(createStripDeadPrototypesPass()); // Remove dead func decls.
  Passes.run(*MClone.get());

  EntryPointsVec NewEntryPoints;
  NewEntryPoints.reserve(Dst.EntryPoints.size());
  // replace entry points with cloned ones
  std::transform(Dst.EntryPoints.cbegin(), Dst.EntryPoints.cend(), std::back_inserter(NewEntryPoints), [&VMap](const Function *F) {
    const Function *F1 = dyn_cast_or_null<Function>(VMap[F]);
    assert(F1 && "clone error");
    return F1;
  });
  assert(!Dst.M && "live module overwritten 1");
  Dst.M = std::move(MClone);
  Dst.EntryPoints = std::move(NewEntryPoints);
}
} // namespace

namespace llvm {
namespace module_split {
void split(std::unique_ptr<Module> M, ModuleDesc /*inout*/&MA, ModuleDesc /*inout*/&MB, StringRef RenameSharedSuff) {
  assert(M && !MA.M && !MB.M);
#ifndef _NDEBUG
  SmallSet<StringRef, 8> EntriesA;
  SmallSet<StringRef, 8> EntriesB;
  std::for_each(MA.EntryPoints.begin(), MA.EntryPoints.end(), [&](const Function *F) {
    EntriesA.insert(F->getName());
  });
  std::for_each(MB.EntryPoints.begin(), MB.EntryPoints.end(), [&](const Function *F) {
    EntriesB.insert(F->getName());
    assert(!EntriesA.contains(F->getName()) && "non-disjoint entry point sets");
  });
#endif // _NDEBUG
  extractCallgraph(*M, MA);
  extractCallgraph(*M, MB);
  M.reset(); // free source module

  if (RenameSharedSuff.empty()) {
    // No renaming requested - done
    return;
  }
  for (GlobalObject &GO : MA.M->global_objects()) {
    if (GO.isDeclaration()) {
      continue;
    }
    StringRef Name = GO.getName();
    auto *F = dyn_cast<Function>(&GO);
    GlobalObject *GO1 = F ? cast_or_null<GlobalObject>(MB.M->getFunction(Name)) : cast_or_null<GlobalObject>(MB.M->getGlobalVariable(Name));

    if (!GO1 || GO1->isDeclaration()) {
      // function or variable is not shared or is a declaration in MB
      continue;
    }
#ifndef _NDEBUG
    if (F) {
      // this is a shared function, must not be an entry point:
      assert(!EntriesA.contains(Name));
      assert(!EntriesB.contains(Name));
    }
#endif // _NDEBUG
    // rename the global object in MB:
    GO1->setName(Name + RenameSharedSuff);
  }
}

void extractCallgraph(const Module &Src, ModuleDesc /*inout*/&M) {
  assert(!M.M && "live module overwritten 2");

  if (M.EntryPoints.empty()) {
    return;
  }
  // Collect all dependencies.
  SetVector<const GlobalValue *> GVs;
  std::vector<const Function *> Workqueue;

  for (const Function *F : M.EntryPoints) {
    GVs.insert(F);
    Workqueue.push_back(F);
  }
  while (!Workqueue.empty()) {
    const Function *F = Workqueue.back();
    Workqueue.pop_back();

    for (const auto &I : instructions(F)) {
      if (const CallBase *CB = dyn_cast<CallBase>(&I))
        if (const Function *CF = CB->getCalledFunction())
          if (!CF->isDeclaration() && !GVs.count(CF)) {
            GVs.insert(CF);
            Workqueue.push_back(CF);
          }
    }
  }
  // It's not easy to trace global variable's uses inside needed functions
  // because global variable can be used inside a combination of operators, so
  // mark all global variables as needed and remove dead ones after
  // cloning.
  for (const auto &G : Src.globals()) {
    GVs.insert(&G);
  }
  cloneSubmoduleInto(Src, GVs, M);
}
} // namespace module_split
} // namespace llvm
