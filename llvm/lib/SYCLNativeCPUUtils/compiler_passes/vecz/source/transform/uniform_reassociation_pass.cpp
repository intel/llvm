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

#include <llvm/ADT/DenseSet.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/Debug.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include "analysis/uniform_value_analysis.h"
#include "analysis/vectorization_unit_analysis.h"
#include "debugging.h"
#include "transform/passes.h"

#define DEBUG_TYPE "vecz"

// WHAT THIS DOES
//
// Where we have some expression involving binary operators over uniform and
// varying values, it can sometimes be advantageous to re-arrange the terms
// to reduce the vectorization overhead. For example, we might have:
//
//   (Varying + Uniform) + Uniform
//
// The above expression requires TWO vector broadcasts of the uniform values,
// and TWO vector additions. However, if we re-associate the operators to get:
//
//  Varying + (Uniform + Uniform)
//
// In this new form, we only need a scalar addition and a single broadcast,
// followed by a single vector addition.
//
// We also make the following transformations:
//
//   (Varying + Uniform) + Varying -> (Varying + Varying) + Uniform
//   Varying + (Varying + Uniform) -> (Varying + Varying) + Uniform
//
// Although these transformations don't reduce the number of vector
// instructions, they may reduce the vector register pressure somewhat. But
// more importantly they may enable further transforms on the CFG.
//
// A common pattern is a conditional statement like this:
//
//    if (uniform_condition && varying_condition) { ... }
//
// Control flow conversion quite often replaces the && with an & in order to
// reduce the number of branches/basic blocks. In this case, however, that is
// counter-productive for us, since we wish to retain the uniform branch and
// linearize the varying one. This pass also splits up such branch conditions.
//
// POTENTIAL FURTHER WORK
//
// Currently, this pass only works on expressions involving a single kind of
// associative and commutative operators. However, similar transformations
// are possible with subtracts and mixtures of subtracts and additions.

using namespace llvm;

namespace {

/// @brief it goes through all the PHI nodes in BB and duplicates the incoming
/// values from "original" to new the new incoming block "extra"
void updatePHIs(BasicBlock &BB, BasicBlock *original, BasicBlock *extra) {
  for (auto &I : BB) {
    auto *const PHI = dyn_cast<PHINode>(&I);
    if (!PHI) {
      break;
    }
    PHI->addIncoming(PHI->getIncomingValueForBlock(original), extra);
  }
}

}  // namespace

namespace vecz {
class Reassociator {
 public:
  Reassociator() {}

  /// @brief perform the Branch Split transformation
  ///
  /// @param[in] F Function to transform.
  /// @param[in] AM FunctionAnalysisManager providing analyses.
  /// @returns true iff any branches were split
  bool run(llvm::Function &F, llvm::FunctionAnalysisManager &AM);

 private:
  /// @brief classification of a binary operand according to whether its
  ///        operands are Uniform, Varying, both (Varying Op Uniform), or non-
  ///        canonically both (i.e. Uniform Op Varying).
  enum class OpForm { Uniform, Varying, Mixed, NonCanonical };

  /// @brief tries to transform a Binary Operator into a canonical form, such
  ///        that if only one operand is Uniform, it is the second operand.
  ///
  /// @param[in] Op the Binary Operator to transform
  /// @returns the form of the canonicalized operator
  OpForm canonicalizeBinOp(llvm::BinaryOperator &Op);

  /// @brief tries to rearrange a binary operator expression to reduce vector
  ///        broadcasts, or to facilitate branch splitting.
  ///
  /// @param[in] Op the Binary Operator to transform
  /// @returns true iff the expression was transformed
  bool reassociate(llvm::BinaryOperator &Op);

  /// @brief canonicalizes a branch into a form that can be split
  ///
  /// @param[in] Branch the branch instruction to canonicalize
  /// @returns true iff the branch condition is mixed (Varying Op Uniform)
  ///          and can be split into two separate branches.
  bool canSplitBranch(llvm::BranchInst &Branch);

  UniformValueResult *UVR = nullptr;
};

Reassociator::OpForm Reassociator::canonicalizeBinOp(llvm::BinaryOperator &Op) {
  if (!UVR->isVarying(&Op)) {
    // Both operands are uniform
    return OpForm::Uniform;
  }

  if (!UVR->isVarying(Op.getOperand(0))) {
    if (Op.isCommutative()) {
      // canonicalize the operator so that operand 1 is uniform
      Op.swapOperands();
      return OpForm::Mixed;
    }
    return OpForm::NonCanonical;
  }

  if (!UVR->isVarying(Op.getOperand(1))) {
    return OpForm::Mixed;
  }

  // Both operands are varying
  return OpForm::Varying;
}

bool Reassociator::reassociate(llvm::BinaryOperator &Op) {
  if (!Op.isAssociative() || !Op.isCommutative()) {
    return false;
  }

  const auto Opcode = Op.getOpcode();
  auto *const LHS = Op.getOperand(0);
  auto *const RHS = Op.getOperand(1);

  auto *const A = dyn_cast<BinaryOperator>(LHS);
  if (A && A->getOpcode() == Opcode && A->hasNUses(1) &&
      canonicalizeBinOp(*A) == OpForm::Mixed) {
    if (UVR->isVarying(RHS)) {
      // Transform (Varying Op Uniform) Op Varying
      // into (Varying Op Varying) Op Uniform
      auto *const P = BinaryOperator::Create(Opcode, A->getOperand(0), RHS,
                                             "varying.reassoc", &Op);
      UVR->setVarying(P);
      Op.setOperand(0, P);
      Op.setOperand(1, A->getOperand(1));
      UVR->remove(A);
      A->eraseFromParent();
      return true;
    } else {
      // Transform (Varying Op Uniform) Op Uniform
      // into Varying Op (Uniform Op Uniform)
      auto *const P = BinaryOperator::Create(Opcode, A->getOperand(1), RHS,
                                             "uniform.reassoc", &Op);
      Op.setOperand(0, A->getOperand(0));
      Op.setOperand(1, P);
      UVR->remove(A);
      A->eraseFromParent();
      return true;
    }
  }

  auto *const B = dyn_cast<BinaryOperator>(RHS);
  if (B && B->getOpcode() == Opcode && B->hasNUses(1) &&
      canonicalizeBinOp(*B) == OpForm::Mixed) {
    // Transform Varying Op (Varying Op Uniform)
    // into (Varying Op Varying) Op Uniform
    auto *const P = BinaryOperator::Create(Opcode, B->getOperand(0), LHS,
                                           "varying.reassoc", &Op);
    Op.setOperand(0, P);
    Op.setOperand(1, B->getOperand(1));
    UVR->setVarying(P);
    UVR->remove(B);
    B->eraseFromParent();
    return true;
  }

  return false;
}

bool Reassociator::canSplitBranch(BranchInst &Branch) {
  if (auto *Op = dyn_cast<BinaryOperator>(Branch.getCondition())) {
    auto Opcode = Op->getOpcode();
    if (Opcode == Instruction::Or || Opcode == Instruction::And) {
      auto Form = canonicalizeBinOp(*Op);
      if (Form == OpForm::Mixed) {
        return true;
      }
    }
  }
  return false;
}

bool Reassociator::run(llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
  auto *DT = &AM.getResult<DominatorTreeAnalysis>(F);
  LoopInfo *LI = nullptr;
  UVR = &AM.getResult<UniformValueAnalysis>(F);

  // Iterate over all instructions in dominance order, so that we always
  // transform an expression before any of its uses.
  SmallVector<BasicBlock *, 16> Blocks;
  DT->getDescendants(&F.getEntryBlock(), Blocks);

  SmallVector<BranchInst *, 4> SplitBranches;
  for (auto *const BB : Blocks) {
    for (auto Iit = BB->begin(); Iit != BB->end();) {
      auto &I = *(Iit++);
      if (auto *BinOp = dyn_cast<BinaryOperator>(&I)) {
        const auto form = canonicalizeBinOp(*BinOp);
        if (form == OpForm::Varying || form == OpForm::Mixed) {
          reassociate(*BinOp);
        }
      } else if (auto *Branch = dyn_cast<BranchInst>(&I)) {
        if (Branch->isConditional() && Branch->getNumSuccessors() == 2 &&
            canSplitBranch(*Branch)) {
          // Lazily obtain the Loop Info
          if (!LI) {
            LI = &AM.getResult<LoopAnalysis>(F);
          }

          if (auto *const L = LI->getLoopFor(BB)) {
            if (L->isLoopExiting(BB)) {
              // No need to do this transform on loop exits (?)
              continue;
            }
          }

          SplitBranches.push_back(Branch);
        }
      }
    }
  }

  if (SplitBranches.empty()) {
    return false;
  }

  auto *PDT = &AM.getResult<PostDominatorTreeAnalysis>(F);

  do {
    auto *Branch = SplitBranches.back();
    SplitBranches.pop_back();
    BasicBlock *BB = Branch->getParent();

    BasicBlock *newBB = SplitBlock(BB, Branch, DT, LI);
    newBB->setName(Twine(BB->getName(), ".cond_split"));

    // update the PostDominatorTree manually..
    PDT->addNewBlock(newBB, PDT->getNode(BB)->getIDom()->getBlock());

    // Remove the unconditional branch created by splitting..
    BB->getTerminator()->eraseFromParent();

    auto *Cond = cast<BinaryOperator>(Branch->getCondition());
    auto *varyingCond = Cond->getOperand(0);
    auto *uniformCond = Cond->getOperand(1);

    // Create a new Uniform branch condition to the Return block..
    // Note that a conditional branch's successors are returned in reverse
    // order, relative to how they appear in the IR, with the "true" target
    // last. However, "getSuccessor(n)" also indexes backwards, from the end.
    auto Opcode = Cond->getOpcode();

    if (Opcode == Instruction::Or) {
      BasicBlock *SuccT = Branch->getSuccessor(0);

      BranchInst::Create(SuccT, newBB, uniformCond, BB);
      Branch->setCondition(varyingCond);

      // If the branch target has PHI nodes, they need to get an extra target
      updatePHIs(*SuccT, newBB, BB);

      // Update Dominator and PostDominator trees..
      DT->insertEdge(BB, SuccT);
      PDT->insertEdge(BB, SuccT);
    } else {
      BasicBlock *SuccF = Branch->getSuccessor(1);

      BranchInst::Create(newBB, SuccF, uniformCond, BB);
      Branch->setCondition(varyingCond);

      // If the branch target has PHI nodes, they need to get an extra target
      updatePHIs(*SuccF, newBB, BB);

      // Update Dominator and PostDominator trees..
      DT->insertEdge(BB, SuccF);
      PDT->insertEdge(BB, SuccF);
    }

    // If we made the condition dead, we can delete it
    if (Cond->use_empty()) {
      Cond->eraseFromParent();
    }

    // The branch may still have a mixed condition after splitting..
    if (canSplitBranch(*Branch)) {
      SplitBranches.push_back(Branch);
    }
  } while (!SplitBranches.empty());

  assert(DT->verify() && "Reassociator: Dominator Tree failed verification");

  assert(PDT->verify() &&
         "Reassociator: Post-Dominator Tree failed verification");

  if (LI) {
    // Unlike the dominator trees, LoopInfo::verify() returns void and asserts
    // internally on failure, for some reason
    LI->verify(*DT);
  }

  return true;
}

/// @brief reassociate uniform binary operators and split branches
PreservedAnalyses UniformReassociationPass::run(Function &F,
                                                FunctionAnalysisManager &AM) {
  Reassociator reassociator;
  const bool changed = reassociator.run(F, AM);
  (void)changed;

  PreservedAnalyses PA;
  PA.preserve<UniformValueAnalysis>();
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<PostDominatorTreeAnalysis>();
  PA.preserve<LoopAnalysis>();
  return PA;
}
}  // namespace vecz
