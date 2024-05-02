// Copyright (C) Codeplay Software Limited
//
// Licensed under the Apache License, Version 2.0 (the "License") with LLVM
// Exceptions; you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/codeplaysoftware/oneapi-construction-kit/blob/main/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <compiler/utils/attributes.h>
#include <compiler/utils/builtin_info.h>
#include <compiler/utils/sub_group_analysis.h>
#include <llvm/ADT/PriorityWorklist.h>
#include <llvm/ADT/SetOperations.h>

using namespace llvm;

namespace compiler {
namespace utils {

GlobalSubgroupInfo::GlobalSubgroupInfo(Module &M, BuiltinInfo &BI) : BI(BI) {
  SmallPtrSet<Function *, 8> UsesSubgroups;
  SmallPriorityWorklist<Function *, 4> Worklist;

  for (auto &F : M) {
    if (F.isDeclaration()) {
      continue;
    }
    auto SGI = std::make_unique<SubgroupInfo>();

    // Assume the 'mux-no-subgroups' attribute is correct. If a pass introduces
    // the use of sub-groups, then it should remove the attribute itself!
    if (hasNoExplicitSubgroups(F)) {
      FunctionMap.insert({&F, std::move(SGI)});
      continue;
    }

    for (auto &BB : F) {
      for (const auto &I : BB) {
        if (auto *const CI = dyn_cast<CallInst>(&I)) {
          if (auto SGBuiltin = isMuxSubgroupBuiltin(CI->getCalledFunction())) {
            // Only add each function to the worklist once
            if (UsesSubgroups.insert(&F).second) {
              Worklist.insert(&F);
            }
            // Track this function's use of this builtin
            SGI->UsedSubgroupBuiltins.insert(SGBuiltin->ID);
          }
        }
      }
    }
    FunctionMap.insert({&F, std::move(SGI)});
  }

  // Collect all functions that contain sub-group calls, including calls to
  // other functions in the module that contain sub-group calls.
  while (!Worklist.empty()) {
    auto *const F = Worklist.pop_back_val();
    const auto &FSubgroups = FunctionMap[F]->UsedSubgroupBuiltins;
    // Track which unique call-graph edges we've traversed, in case F ends up
    // calling the same function multiple times. The set of builtins used by
    // this item isn't going to change while we're working on it.
    SmallPtrSet<Function *, 4> AlreadyUnioned;
    for (auto *const U : F->users()) {
      if (auto *const CI = dyn_cast<CallInst>(U)) {
        auto *const CallerF = CI->getFunction();
        // If we haven't seen this function before, we need to process it and
        // propagate its users.
        if (UsesSubgroups.insert(CallerF).second) {
          Worklist.insert(CallerF);
        }
        // If we've recorded that CallerF calls F for the first time in this
        // loop, CallerF's set of used builtins gains all the builtins used by
        // F.
        if (AlreadyUnioned.insert(CallerF).second) {
          auto &CallerSubgroups = FunctionMap[CallerF]->UsedSubgroupBuiltins;
          // If the set union produces a new set...
          if (set_union(CallerSubgroups, FSubgroups)) {
            // ... we might have previously visited CallerF when it had fewer
            // registered uses of sub-groups. Thus we need to stick it back on
            // the worklist to propagate these to its users.
            Worklist.insert(CallerF);
          }
        }
      }
    }
  }
}

bool GlobalSubgroupInfo::usesSubgroups(const llvm::Function &F) const {
  auto I = FunctionMap.find(&F);
  assert(I != FunctionMap.end() && "Missing entry for function");
  return !I->second->UsedSubgroupBuiltins.empty();
}

std::optional<Builtin> GlobalSubgroupInfo::isMuxSubgroupBuiltin(
    const Function *F) const {
  if (!F) {
    return std::nullopt;
  }
  auto SGBuiltin = BI.analyzeBuiltin(*F);

  switch (SGBuiltin.ID) {
    default:
      break;
    case eMuxBuiltinSubGroupBarrier:
    case eMuxBuiltinGetSubGroupSize:
    case eMuxBuiltinGetMaxSubGroupSize:
    case eMuxBuiltinGetNumSubGroups:
    case eMuxBuiltinGetSubGroupId:
    case eMuxBuiltinGetSubGroupLocalId:
      return SGBuiltin;
  }

  if (auto GroupOp = BI.isMuxGroupCollective(SGBuiltin.ID);
      GroupOp && GroupOp->isSubGroupScope()) {
    return SGBuiltin;
  }

  return std::nullopt;
}

AnalysisKey SubgroupAnalysis::Key;

SubgroupAnalysis::Result SubgroupAnalysis::run(Module &M,
                                               ModuleAnalysisManager &AM) {
  return GlobalSubgroupInfo(M, AM.getResult<BuiltinInfoAnalysis>(M));
}

PreservedAnalyses SubgroupAnalysisPrinterPass::run(Module &M,
                                                   ModuleAnalysisManager &AM) {
  const auto &Info = AM.getResult<SubgroupAnalysis>(M);

  for (auto &F : M) {
    if (F.isDeclaration()) {
      continue;
    }
    OS << "Function '" << F.getName() << "' uses";
    if (!Info.usesSubgroups(F)) {
      OS << " no sub-group builtins\n";
      continue;
    }
    auto *FInfo = Info[&F];
    assert(FInfo && "Missing function info");
    const auto &UsedBuiltins = FInfo->UsedSubgroupBuiltins;
    // Note: this output isn't stable and shouldn't be relied upon. It's mostly
    // for developer analysis.
    OS << " " << UsedBuiltins.size() << " sub-group builtin"
       << (UsedBuiltins.size() == 1 ? "" : "s") << ": "
       << static_cast<unsigned>(*UsedBuiltins.begin());
    for (auto B :
         make_range(std::next(UsedBuiltins.begin()), UsedBuiltins.end())) {
      OS << "," << static_cast<unsigned>(B);
    }
    OS << "\n";
  }

  return PreservedAnalyses::all();
}
}  // namespace utils
}  // namespace compiler
