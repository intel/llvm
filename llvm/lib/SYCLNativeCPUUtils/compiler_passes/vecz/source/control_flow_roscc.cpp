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

#include "control_flow_roscc.h"

#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/Support/Debug.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include "analysis/uniform_value_analysis.h"
#include "debugging.h"
#include "ir_cleanup.h"

#define DEBUG_TYPE "vecz-cf"

// WHAT THIS DOES
//
// A common pattern in OpenCL kernels is a line near the start of the program
// like the following:
//
//    if (some_condition) return;
//
// Where "some_condition" is non-uniform, the BOSCC control flow optimization
// can do very well with this. However, without BOSCC, the entire program will
// have been linearized and the early return will disappear entirely. It is
// desirable to maintain this sort of early exit branch in order to avoid
// doing unnecessary work. We can do this by inserting a uniform branch to the
// return block without the need to duplicate the rest of the kernel into
// uniform and non-uniform versions, as BOSCC does. This can improve the
// performance significantly without requiring complex CFG changes.

using namespace llvm;
using namespace vecz;

namespace {
/// @brief checks if the given block contains only a return instruction
bool isReturnBlock(const llvm::BasicBlock &BB) {
  if (BB.size() != 1) {
    return false;
  }

  auto *T = BB.getTerminator();
  if (auto *const branch = dyn_cast<BranchInst>(T)) {
    if (branch->isUnconditional()) {
      // We can see straight through a block that only contains a single
      // unconditional branch.
      return isReturnBlock(*branch->getSuccessor(0));
    }
  }

  return isa<ReturnInst>(T);
}
}  // namespace

bool ControlFlowConversionState::ROSCCGadget::run(Function &F) {
  bool changed = false;

  SmallVector<BranchInst *, 4> RetBranches;
  for (auto &BB : F) {
    if (LI->getLoopFor(&BB)) {
      // No need to do this transform on loop exits
      continue;
    }

    auto *T = BB.getTerminator();
    if (auto *Branch = dyn_cast<BranchInst>(T)) {
      if (Branch->isConditional() && Branch->getNumSuccessors() == 2) {
        Value *cond = Branch->getCondition();
        if (UVR->isVarying(cond)) {
          size_t countReturns = 0;
          for (auto *succ : Branch->successors()) {
            if (isReturnBlock(*succ)) {
              ++countReturns;
            }
          }

          // Only consider ROSCC when there is exactly one returning successor.
          if (countReturns == 1) {
            RetBranches.push_back(Branch);
          }
        }
      }
    }
  }

  ConstantInt *trueCI = ConstantInt::getTrue(F.getContext());
  ConstantInt *falseCI = ConstantInt::getFalse(F.getContext());

  for (auto *Branch : RetBranches) {
    BasicBlock *BB = Branch->getParent();

    BasicBlock *newBB = SplitBlock(BB, Branch, DT, LI);
    newBB->setName(Twine(BB->getName(), ".ROSCC"));

    // update the PostDominatorTree manually..
    auto *Node = PDT->getNode(BB);
    assert(Node && "Could not get node");
    auto *IDom = Node->getIDom();
    assert(IDom && "Could not get IDom");
    auto *Block = IDom->getBlock();
    assert(Block && "Could not get Block");
    PDT->addNewBlock(newBB, Block);

    // Remove the unconditional branch created by splitting..
    IRCleanup::deleteInstructionNow(BB->getTerminator());

    // Create a new Uniform branch condition to the Return block..
    // Note that a conditional branch's successors are returned in reverse
    // order, relative to how they appear in the IR, with the "true" target
    // last. However, "getSuccessor(n)" also indexes backwards, from the end.
    BasicBlock *SuccT = Branch->getSuccessor(0);
    BasicBlock *SuccF = Branch->getSuccessor(1);
    const bool Which = isReturnBlock(*SuccT);

    BasicBlock *ReturnBlock = Which ? SuccT : SuccF;
    Value *Cond = Branch->getCondition();
    ICmpInst *newCond =
        new ICmpInst(*BB, CmpInst::ICMP_EQ, Cond, Which ? falseCI : trueCI);
    newCond->setName(Twine(Cond->getName(), ".ROSCC"));
    BranchInst::Create(newBB, ReturnBlock, newCond, BB);

    // Update Dominator and PostDominator trees..
    DT->insertEdge(BB, ReturnBlock);
    PDT->insertEdge(BB, ReturnBlock);

    changed = true;
  }

  assert((!changed || DT->verify()) &&
         "ROSCC: Dominator Tree failed verification");

  assert((!changed || PDT->verify()) &&
         "ROSCC: Post-Dominator Tree failed verification");

  return changed;
}
