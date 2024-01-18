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
//
// Implementation based on Section 5.2 of the paper:
// Florian Brandner, Benoit Boissinot, Alain Darte, Benoît Dupont de Dinechin,
// Fabrice Rastello.
// Computing Liveness Sets for SSA-Form Programs.
// [Research Report] RR-7503, INRIA. 2011, pp.25. inria-00558509v2
//
// https://hal.inria.fr/inria-00558509v2

#include "analysis/liveness_analysis.h"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Analysis/Passes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Instructions.h>

#include "vectorization_unit.h"

using namespace llvm;
using namespace vecz;

llvm::AnalysisKey LivenessAnalysis::Key;

namespace {

// Returns true if V defines a variable and is likely to require a register
bool definesVariable(const Value &V) {
  // Constants are likely to be immediate values
  if (isa<Constant>(V)) {
    return false;
  }

  // If a value isn't used, it can't be live
  if (V.user_empty()) {
    return false;
  }

  const auto valueType = V.getType();
  return !valueType->isVoidTy() && !valueType->isLabelTy() &&
         !valueType->isTokenTy() && !valueType->isMetadataTy();
}

// Tries to push a value onto the set, if it is not there already.
// Returns true if the value was pushed, false otherwise.
//
// Note that since the implementation completely processes every instruction
// sequentially, only the last element needs to be checked.
inline bool pushOnce(BlockLivenessInfo::LiveSet &s, Value *V) {
  if (!s.empty() && s.back() == V) {
    return false;
  }
  s.push_back(V);
  return true;
}

}  // namespace

class LivenessResult::Impl {
 public:
  Impl(LivenessResult &lr) : LR(lr) {}

  void recalculate();

 private:
  LivenessResult &LR;

  void computeByVar(const BasicBlock &BB);

  void computeVar(Value *V, const BasicBlock *BB);

  void mark(Value *V, const BasicBlock *parent, const BasicBlock *BB);

  void calculateMaxRegistersInBlock(const llvm::BasicBlock *BB);

  // private utility method for code conciseness
  inline BlockLivenessInfo &info(const BasicBlock *BB) const {
    auto BIi = LR.BlockInfos.find(BB);
    assert(BIi != LR.BlockInfos.end() && "Block Liveness Info does not exist!");
    return BIi->second;
  }
};

LivenessResult LivenessAnalysis::run(llvm::Function &F,
                                     llvm::FunctionAnalysisManager &) {
  Result R(F);
  R.recalculate();
  return R;
}

size_t LivenessResult::getMaxLiveVirtualRegisters() const {
  return maxNumberOfLiveValues;
}

const BlockLivenessInfo &LivenessResult::getBlockInfo(
    const BasicBlock *BB) const {
  auto found = BlockInfos.find(BB);
  assert(found != BlockInfos.end() && "No liveness information for BasicBlock");
  return found->second;
}

void LivenessResult::recalculate() {
  maxNumberOfLiveValues = 0;

  BlockInfos.clear();

  Impl impl(*this);
  impl.recalculate();
}

void LivenessResult::Impl::recalculate() {
  auto &F = LR.F;

  // Create infos in advance so things don't relocate under our feet.
  for (auto &BB : F) {
    (void)LR.BlockInfos[&BB];
  }

  // Arguments are always live-ins of the entry block (if they are used).
  {
    auto *BB = &F.getEntryBlock();
    auto &BI = info(BB);
    for (auto &arg : F.args()) {
      if (!arg.use_empty()) {
        BI.LiveIn.push_back(&arg);
        computeVar(&arg, BB);
      }
    }
  }

  // Add all other variables to the live sets.
  for (auto &BB : F) {
    auto &BI = LR.BlockInfos[&BB];
    for (auto &I : BB) {
      if (definesVariable(I)) {
        if (isa<PHINode>(I)) {
          // PHI nodes are always live-ins.
          BI.LiveIn.push_back(&I);
        }
        computeVar(&I, &BB);
      }
    }
  }

  // Calculate the maximum number of live values in every block.
  for (auto &BB : F) {
    calculateMaxRegistersInBlock(&BB);
  }

  // Store the largest number of live values in the function.
  for (const auto &entry : LR.BlockInfos) {
    LR.maxNumberOfLiveValues = std::max(LR.maxNumberOfLiveValues,
                                        entry.getSecond().MaxRegistersInBlock);
  }
}

void LivenessResult::Impl::computeVar(Value *V, const BasicBlock *BB) {
  SmallPtrSet<const BasicBlock *, 8> UseBlocks;
  for (auto *User : V->users()) {
    if (auto *UI = dyn_cast<Instruction>(User)) {
      if (auto *PHI = dyn_cast<PHINode>(UI)) {
        for (unsigned i = 0, n = PHI->getNumIncomingValues(); i != n; ++i) {
          if (PHI->getIncomingValue(i) == V) {
            const auto *Incoming = PHI->getIncomingBlock(i);

            if (pushOnce(info(Incoming).LiveOut, V) && Incoming != BB) {
              UseBlocks.insert(Incoming);
            }
          }
        }
      } else {
        const auto *Parent = UI->getParent();
        if (Parent != BB) {
          UseBlocks.insert(Parent);
        }
      }
    }
  }

  for (auto *UB : UseBlocks) {
    if (pushOnce(info(UB).LiveIn, V)) {
      mark(V, BB, UB);
    }
  }
}

void LivenessResult::Impl::mark(Value *V, const BasicBlock *parent,
                                const BasicBlock *BB) {
  // Propagate backward
  for (const auto *pred : predecessors(BB)) {
    auto &PBI = info(pred);
    if (pushOnce(PBI.LiveOut, V) && pred != parent && pushOnce(PBI.LiveIn, V)) {
      mark(V, parent, pred);
    }
  }
}

void LivenessResult::Impl::calculateMaxRegistersInBlock(const BasicBlock *BB) {
  auto &BI = LR.BlockInfos[BB];
  const SmallPtrSet<const Value *, 16> liveOut(BI.LiveOut.begin(),
                                               BI.LiveOut.end());
  SmallPtrSet<const Value *, 16> seenButNotInLiveOut;

  auto maxRegistersUsed = liveOut.size();
  auto registersUsed = liveOut.size();

  // Walk backwards through instructions in a block to count the maximum number
  // of live values in that block.
  for (auto &inst : make_range(BB->rbegin(), BB->rend())) {
    // Phi nodes were in live out or were counted as operands. No need to
    // decrement the registerCount, as one of the arguments used a register.
    if (isa<PHINode>(&inst)) {
      break;
    }

    // Operands are live so they use a register. Increment registerCount if not
    // in live out or already counted.
    for (const auto *operand : inst.operand_values()) {
      if (definesVariable(*operand) && !liveOut.count(operand) &&
          !seenButNotInLiveOut.count(operand)) {
        registersUsed++;
        seenButNotInLiveOut.insert(operand);
      }
    }

    // If inst defines a variable, one less register was used before it
    if (definesVariable(inst)) {
      registersUsed--;
    }

    maxRegistersUsed = std::max(registersUsed, maxRegistersUsed);
  }

  assert(registersUsed == BI.LiveIn.size() &&
         "Final number of live values inconsistent with live-in");

  BI.MaxRegistersInBlock = maxRegistersUsed;
}
