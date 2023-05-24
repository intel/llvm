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

#include <llvm/IR/Dominators.h>
#include <llvm/Transforms/Scalar/LoopPassManager.h>

#include <unordered_set>

#include "debugging.h"
#include "multi_llvm/multi_llvm.h"
#include "transform/passes.h"

using namespace llvm;

PreservedAnalyses vecz::SimplifyInfiniteLoopPass::run(
    Loop &L, LoopAnalysisManager &, LoopStandardAnalysisResults &AR,
    LPMUpdater &) {
  bool modified = false;

  SmallVector<BasicBlock *, 1> loopExitBlocks;
  L.getExitBlocks(loopExitBlocks);

  // If we have an infinite loop, create a virtual exit block that will target
  // the unique exit block of the function.
  if (loopExitBlocks.empty()) {
    BasicBlock *latch = L.getLoopLatch();
    assert(latch && "Loop should have a unique latch.");

    Function *F = L.getHeader()->getParent();

    // Get the return block of the function.
    std::vector<BasicBlock *> returnBlocks;
    for (BasicBlock &BB : *F) {
      if (isa<ReturnInst>(BB.getTerminator())) {
        returnBlocks.push_back(&BB);
      }
    }

    if (returnBlocks.empty() || returnBlocks.size() > 1) {
      assert(false && "Function should have only one exit.");
      return PreservedAnalyses::all();
    }

    // The target of the virtual exit block of the infinite loop.
    BasicBlock *target = returnBlocks[0];

    // Replace the terminator of the latch with a fake conditional branch that
    // will actually always target the header to maintain the semantic of the
    // program.
    latch->getTerminator()->eraseFromParent();
    AR.DT.deleteEdge(latch, L.getHeader());
    BasicBlock *virtualExit =
        BasicBlock::Create(F->getContext(), L.getName() + ".virtual_exit", F);
    AR.DT.addNewBlock(virtualExit, latch);
    BranchInst::Create(L.getHeader(), virtualExit,
                       ConstantInt::getTrue(F->getContext()), latch);
    AR.DT.insertEdge(latch, L.getHeader());
    AR.DT.insertEdge(latch, virtualExit);
    BranchInst::Create(target, virtualExit);
    AR.DT.insertEdge(virtualExit, target);

    assert(AR.DT.verify() &&
           "SimplifyInfiniteLoopPass: Dominator Tree failed verification");

    std::unordered_set<Instruction *> toBlend;
    // Find all instructions used in `target` that may be defined after the
    // infinite loop, for which adding the edge from the infinite loop to the
    // return block may break the SSA form.
    for (Instruction &I : *target) {
      if (!isa<PHINode>(&I)) {
        for (Value *op : I.operands()) {
          if (Instruction *opI = dyn_cast<Instruction>(op)) {
            if (opI->getParent() != target) {
              toBlend.insert(opI);
            }
          }
        }
      }
    }

    // Update the phi nodes in the return block because we added a new
    // predecessor to it.
    for (Instruction &I : *target) {
      if (auto *PHI = dyn_cast<PHINode>(&I)) {
        PHI->addIncoming(UndefValue::get(PHI->getType()), virtualExit);
      }
    }
    // Add new phi nodes for instructions computed in `toBlend`.
    for (Instruction *I : toBlend) {
      PHINode *PHI = PHINode::Create(I->getType(), 2, I->getName() + ".blend",
                                     &target->front());
      for (BasicBlock *pred : predecessors(target)) {
        if (pred != virtualExit) {
          PHI->addIncoming(I, pred);
        } else {
          PHI->addIncoming(UndefValue::get(PHI->getType()), pred);
        }
      }
    }

    modified = true;
  } else if (loopExitBlocks.size() == 1) {
    // Canonicalize any other infinite loops so that the loop header is the
    // true condition successor.
    auto *const latch = L.getLoopLatch();
    auto *const header = L.getHeader();
    auto *const T = latch->getTerminator();
    if (auto *const branch = dyn_cast<BranchInst>(T)) {
      if (branch->isConditional()) {
        if (auto *const cond = dyn_cast<Constant>(branch->getCondition())) {
          if (branch->getSuccessor(1) == header) {
            modified = true;
            auto &ctx = latch->getParent()->getContext();
            branch->setCondition(cond->isOneValue()
                                     ? ConstantInt::getFalse(ctx)
                                     : ConstantInt::getTrue(ctx));
            branch->swapSuccessors();
          }
        }
      }
    }
  }

  if (!modified) {
    return PreservedAnalyses::all();
  }

  return getLoopPassPreservedAnalyses();
}
