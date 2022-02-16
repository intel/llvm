//===-- SYCLStripDeadDebugInfo.cpp - Strip debug info from split module ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SYCLStripDeadDebugInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"

#include <set>
#include <unordered_set>

using namespace llvm;

namespace {

// The implementation of this function is almost the same as of the one for
// llvm::StripDeadDebugInfoPass, but some conditions for delete debug info more
// aggressive taking into account that processed module is split from another
// bigger module and contains only a subset of functions from it.
bool stripDeadDebugInfoImpl(Module &M) {
  bool Changed = false;

  LLVMContext &C = M.getContext();

  // Find all debug info in F. This is actually overkill in terms of what we
  // want to do, but we want to try and be as resilient as possible in the face
  // of potential debug info changes by using the formal interfaces given to us
  // as much as possible.
  DebugInfoFinder F;
  F.processModule(M);

  // For each compile unit, find the live set of global variables/functions and
  // replace the current list of potentially dead global variables/functions
  // with the live list.
  SmallVector<Metadata *, 64> LiveGlobalVariables;
  DenseSet<DIGlobalVariableExpression *> VisitedSet;

  std::set<DIGlobalVariableExpression *> LiveGVs;
  for (GlobalVariable &GV : M.globals()) {
    SmallVector<DIGlobalVariableExpression *, 1> GVEs;
    GV.getDebugInfo(GVEs);
    for (auto *GVE : GVEs)
      LiveGVs.insert(GVE);
  }

  std::unordered_set<const DISubprogram *> PresentFunctionsSPs;
  if (AggressiveMode) {
    for (auto &Func : M.functions())
      if (auto *FSP = cast_or_null<DISubprogram>(Func.getSubprogram()))
        PresentFunctionsSPs.insert(FSP);
  }

  std::set<DICompileUnit *> LiveCUs;
  // Any CU referenced from a subprogram is live.
  // Diff 1: Additional condition which is added here in contrast to initial
  // pass is that this debug info should link to a function in a processed
  // module.
  for (const DISubprogram *SP : F.subprograms())
    if (SP->getUnit() && (!AggressiveMode || PresentFunctionsSPs.count(SP)))
      LiveCUs.insert(SP->getUnit());

  bool HasDeadCUs = false;
  for (DICompileUnit *DIC : F.compile_units()) {
    // Create our live global variable list.
    bool GlobalVariableChange = false;
    for (auto *DIG : DIC->getGlobalVariables()) {
      // Diff 2: Second difference from initial pass is that we don't include
      // additional compile units with constant global expressions because if
      // a global constant is used in a function then a compile unit with that
      // function should be already added.
      if (!AggressiveMode && DIG->getExpression() &&
          DIG->getExpression()->isConstant())
        LiveGVs.insert(DIG);

      // Make sure we only visit each global variable only once.
      if (!VisitedSet.insert(DIG).second)
        continue;

      // If a global variable references DIG, the global variable is live.
      if (LiveGVs.count(DIG))
        LiveGlobalVariables.push_back(DIG);
      else
        GlobalVariableChange = true;
    }

    if (!LiveGlobalVariables.empty())
      LiveCUs.insert(DIC);
    else if (!LiveCUs.count(DIC))
      HasDeadCUs = true;

    // If we found dead global variables, replace the current global
    // variable list with our new live global variable list.
    if (GlobalVariableChange) {
      DIC->replaceGlobalVariables(MDTuple::get(C, LiveGlobalVariables));
      Changed = true;
    }

    // Reset lists for the next iteration.
    LiveGlobalVariables.clear();
  }

  if (HasDeadCUs) {
    // Delete the old node and replace it with a new one
    NamedMDNode *NMD = M.getOrInsertNamedMetadata("llvm.dbg.cu");
    NMD->clearOperands();
    if (!LiveCUs.empty())
      for (DICompileUnit *CU : LiveCUs)
        NMD->addOperand(CU);
    Changed = true;
  }

  return Changed;
}

} // anonymous namespace

namespace llvm {

PreservedAnalyses SYCLStripDeadDebugInfo::run(Module &M,
                                              ModuleAnalysisManager &AM) {
  stripDeadDebugInfoImpl(M);
  return PreservedAnalyses::all();
}

} // namespace llvm
