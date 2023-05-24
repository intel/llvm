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

#include "transform/common_gep_elimination_pass.h"

#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/Dominators.h>

#include <unordered_map>

#include "analysis/control_flow_analysis.h"
#include "analysis/divergence_analysis.h"
#include "analysis/vectorization_unit_analysis.h"
#include "debugging.h"
#include "ir_cleanup.h"
#include "vectorization_unit.h"

using namespace llvm;
using namespace vecz;

char CommonGEPEliminationPass::PassID = 0;

PreservedAnalyses CommonGEPEliminationPass::run(Function &F,
                                                FunctionAnalysisManager &AM) {
  const DominatorTree &DT = AM.getResult<DominatorTreeAnalysis>(F);

  // Redundant GEPs to remove
  SmallPtrSet<GetElementPtrInst *, 16> toDelete;
  // GEPs we come across.
  std::unordered_multimap<Value *, GetElementPtrInst *> GEPs;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
        Value *Ptr = GEP->getPointerOperand();
        // If this is the first time we meet the source of the GEP, just add
        // it to the multimap and look for another GEP.
        if (GEPs.find(Ptr) == GEPs.end()) {
          GEPs.emplace(Ptr, GEP);
          continue;
        }

        // The range of values that have the key `Ptr`.
        auto Range = GEPs.equal_range(Ptr);
        auto it = Range.first;
        for (; it != Range.second; it++) {
          auto *trackedGEP = it->second;
          if (GEP->getNumIndices() != trackedGEP->getNumIndices()) {
            continue;
          }

          // With opaque pointers, we need to check the element types as well.
          if (GEP->getSourceElementType() !=
              trackedGEP->getSourceElementType()) {
            continue;
          }

          unsigned i = 0;
          for (; i < GEP->getNumIndices(); i++) {
            Value *lhs = GEP->getOperand(i + 1);
            Value *rhs = trackedGEP->getOperand(i + 1);

            // Both GEPs we compare are not the same, stop comparing.
            if (lhs != rhs) {
              break;
            }
          }

          // trackedGEP does the same operation as GEP, so replace GEP
          // with the already tracked GEP.
          if (i == GEP->getNumIndices()) {
            if (DT.dominates(trackedGEP->getParent(), GEP->getParent())) {
              GEP->replaceAllUsesWith(trackedGEP);
              toDelete.insert(GEP);
              break;
            }
          }
        }
        // We iterated over all values whose key is Ptr, but haven't found
        // a matching GEP, so add the latter to the multimap.
        if (it == Range.second) {
          GEPs.emplace(Ptr, GEP);
        }
      }
    }
  }

  // Proceed to remove every duplicate GEP we found.
  for (auto *GEP : toDelete) {
    IRCleanup::deleteInstructionNow(GEP);
  }

  PreservedAnalyses Preserved;
  Preserved.preserve<DominatorTreeAnalysis>();
  Preserved.preserve<LoopAnalysis>();
  Preserved.preserve<CFGAnalysis>();
  Preserved.preserve<DivergenceAnalysis>();

  return Preserved;
}
