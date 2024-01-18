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
// This pass aims to optimize the CFG by hoisting instructions out of triangle
// or diamond patterns (i.e. "if" or "if..else" constructs) where it determines
// that executing all the instructions in all branch targets is cheaper than
// actually branching. This is especially the case when BOSCC is active as the
// BOSCC gadget introduces potentially-expensive AND/OR reduction operations
// in order to branch to the uniform version of each Basic Block. To such end,
// the pass needs to use the Uniform Value Analysis result, since only varying
// branch conditions will be affected by BOSCC in such a way. We also need
// access to the Target Transform Info result from the Vectorization Unit in
// order to make target-dependent cost-based decisions.
//
// This pass only hoists instructions out of conditional blocks, and does not
// directly modify the CFG, so it is intended that CFG Simplification Pass to
// be run afterwards, in order to eliminate the now-redundant Basic Blocks and
// transform PHI nodes into select instructions. Therefore, the
// pre-linearization pass is implemented as an llvm::FunctionPass so it can
// be run in the middle of the Vecz Preparation Pass.
//
// Pre-Linearization is currently unable to hoist memory operations, since
// doing so will require the correct masked versions to be generated, which
// would require a lot of special extra handling.

#include <llvm/ADT/DepthFirstIterator.h>
#include <llvm/ADT/GraphTraits.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Pass.h>
#include <llvm/Support/InstructionCost.h>
#include <llvm/Transforms/Utils/LoopUtils.h>
#include <multi_llvm/multi_llvm.h>
#include <multi_llvm/vector_type_helper.h>

#include "analysis/uniform_value_analysis.h"
#include "analysis/vectorization_unit_analysis.h"
#include "debugging.h"
#include "transform/passes.h"
#include "vectorization_unit.h"
#include "vecz/vecz_choices.h"

using namespace llvm;
using namespace vecz;

namespace {
bool isTrivialBlock(const llvm::BasicBlock &BB) {
  for (const auto &I : BB) {
    if (I.mayReadOrWriteMemory() || I.mayHaveSideEffects() ||
        llvm::isa<llvm::PHINode>(&I)) {
      return false;
    }
  }
  return true;
}

// This is an estimate of the cycle count for executing the entire block,
// not including the terminating branch instruction, obtained by summing
// the cost (Reciprocal Throughput) of each individual instruction.
// This assumes sequential execution (no Instruction Level Parallelism)
// and takes no account of Data Hazards &c so is not guaranteed to be
// entirely accurate.
InstructionCost calculateBlockCost(const BasicBlock &BB,
                                   const TargetTransformInfo &TTI) {
  InstructionCost cost;
  for (const auto &I : BB) {
    if (I.isTerminator()) {
      break;
    }

    InstructionCost inst_cost =
        TTI.getInstructionCost(&I, TargetTransformInfo::TCK_RecipThroughput);

    // When a vector instruction is encountered, we multiply by the vector
    // width, because it will either be scalarized into that many individual
    // instructions during scalarization, or packetized by duplication.
    // This works on the assumption that throughput does not depend on the
    // vector width. This calculation may need refining in future.
    if (I.getType()->isVectorTy()) {
      inst_cost *= multi_llvm::getVectorNumElements(I.getType());
    }

    cost += inst_cost;
  }
  return cost;
}

// It creates a temporary function in order to build a target-dependent
// vector AND reduction inside it, in order to calculate the cost of it.
InstructionCost calculateBoolReductionCost(LLVMContext &context, Module *module,
                                           const TargetTransformInfo &TTI,
                                           llvm::ElementCount width) {
  Type *cond_ty = VectorType::get(Type::getInt1Ty(context), width);

  FunctionType *new_fty =
      FunctionType::get(Type::getVoidTy(context), {cond_ty}, false);

  // LLVM 11 requires the function to be in a valid (existing) module in
  // order to create a simple vector reduction with the specified opcode.
  auto *F = Function::Create(new_fty, Function::InternalLinkage, "tmp", module);
  auto *BB = BasicBlock::Create(context, "reduce", F);
  IRBuilder<> B(BB);
  multi_llvm::createSimpleTargetReduction(B, &TTI, &*F->arg_begin(),
                                          RecurKind::And);
  const InstructionCost cost = calculateBlockCost(*BB, TTI);

  // We don't really need that function in the module anymore because it's
  // only purpose was to be used for analysis, so we go ahead and remove it.
  F->removeFromParent();
  delete F;
  return cost;
}

bool hoistInstructions(BasicBlock &BB, BranchInst &Branch, bool exceptions) {
  const auto &DL = BB.getModule()->getDataLayout();
  const bool TrueBranch = (Branch.getSuccessor(0) == &BB);
  DenseMap<Value *, Value *> safeDivisors;

  bool modified = false;
  while (!BB.front().isTerminator()) {
    auto &I = BB.front();
    I.moveBefore(&Branch);
    modified = true;

    if (!exceptions) {
      // we don't need to mask division operations if they don't trap
      continue;
    }

    if (!isa<BinaryOperator>(&I)) {
      // we only hoist binary operators
      continue;
    }
    auto *binOp = cast<BinaryOperator>(&I);
    // It is potentially dangerous to hoist division operations, since
    // the RHS could be zero or INT_MIN on some lanes, unless it's a
    // constant.
    bool isUnsigned = false;
    switch (binOp->getOpcode()) {
      default:
        break;
      case Instruction::UDiv:
      case Instruction::URem:
        isUnsigned = true;
        LLVM_FALLTHROUGH;
      case Instruction::SDiv:
      case Instruction::SRem: {
        auto *divisor = binOp->getOperand(1);
        if (auto *C = dyn_cast<Constant>(divisor)) {
          if (C->isZeroValue()) {
            // Divides by constant zero can be a NOP since there is no
            // division by zero exception in OpenCL.
            I.replaceAllUsesWith(binOp->getOperand(0));
            I.eraseFromParent();
          }
        } else {
          // if the divisor could be illegal, we need to guard it with a
          // select instruction generated from the branch condition.
          auto &masked = safeDivisors[divisor];
          if (!masked) {
            // NOTE this function does not check for the pattern
            // "select (x eq 0) 1, x" or equivalent, so we might want to
            // write it ourselves, but Instruction Combining cleans it
            // up. NOTE that for a signed division, we also have to
            // consider the potential overflow situation, which is not
            // so simple
            if (isUnsigned && isKnownNonZero(divisor, DL)) {
              // Static analysis concluded it can't be zero, so we don't
              // need to do anything.
              masked = divisor;
            } else {
              Value *one = ConstantInt::get(divisor->getType(), 1);
              Value *cond = Branch.getCondition();

              if (TrueBranch) {
                masked =
                    SelectInst::Create(cond, divisor, one,
                                       divisor->getName() + ".hoist_guard", &I);
              } else {
                masked =
                    SelectInst::Create(cond, one, divisor,
                                       divisor->getName() + ".hoist_guard", &I);
              }
            }
          }

          if (masked != divisor) {
            binOp->setOperand(1, masked);
          }
        }
      } break;
    }
  }
  return modified;
}
}  // namespace

PreservedAnalyses PreLinearizePass::run(Function &F,
                                        FunctionAnalysisManager &AM) {
  VectorizationUnitAnalysis::Result R =
      AM.getResult<VectorizationUnitAnalysis>(F);
  const TargetTransformInfo &TTI = AM.getResult<TargetIRAnalysis>(F);
  const VectorizationUnit &VU = R.getVU();

  bool modified = false;
  auto &LI = AM.getResult<LoopAnalysis>(F);
  const bool div_exceptions =
      VU.choices().isEnabled(VectorizationChoices::eDivisionExceptions);

  InstructionCost boscc_cost;
  UniformValueResult *UVR = nullptr;
  if (VU.choices().linearizeBOSCC()) {
    boscc_cost = calculateBoolReductionCost(F.getContext(), F.getParent(), TTI,
                                            VU.width());
    UVR = &AM.getResult<UniformValueAnalysis>(F);
  }

  auto dfo = depth_first(&F.getEntryBlock());
  SmallVector<BasicBlock *, 16> blocks(dfo.begin(), dfo.end());

  DenseMap<BasicBlock *, BasicBlock *> single_succs;
  for (auto *BB : blocks) {
    single_succs[BB] = BB->getSingleSuccessor();
  }

  for (auto BBit = blocks.rbegin(), BBe = blocks.rend(); BBit != BBe; ++BBit) {
    BasicBlock *BB = *BBit;

    // Check that all hoistable successor blocks are in the same loop
    Loop *block_loop = LI.getLoopFor(BB);

    if (succ_size(BB) >= 2) {
      bool simple = true;
      SmallPtrSet<BasicBlock *, 2> targets;
      for (auto *succ : successors(BB)) {
        if (BasicBlock *target = single_succs[succ]) {
          targets.insert(target);
        }
      }

      SmallVector<BasicBlock *, 2> hoistable;
      SmallPtrSet<BasicBlock *, 2> new_succs;
      for (auto *succ : successors(BB)) {
        if (targets.count(succ) == 0) {
          if (single_succs[succ] == nullptr || pred_size(succ) != 1 ||
              LI.getLoopFor(succ) != block_loop || !isTrivialBlock(*succ)) {
            simple = false;
            break;
          }
          hoistable.push_back(succ);
        } else {
          // these "bypass" successors are going to stay where they are
          new_succs.insert(succ);
        }
      }
      if (!simple || hoistable.empty()) {
        continue;
      }

      // The cost of a "bypass" branch is essentially zero. This occurs in a
      // "triangle" type control struct (i.e. if with no else).
      InstructionCost min_cost = new_succs.empty() ? InstructionCost::getMax()
                                                   : InstructionCost::getMin();

      // The total cost of executing every successor sequentially
      InstructionCost total_cost = 0;

      for (auto *succ : hoistable) {
        const InstructionCost block_cost = calculateBlockCost(*succ, TTI);
        if (block_cost < min_cost) {
          min_cost = block_cost;
        }
        total_cost += block_cost;
        new_succs.insert(single_succs[succ]);
      }

      // One of the successors was going to get executed anyway, so we can
      // discount the cost of the cheapest one from the total cost.
      total_cost -= min_cost;

      // The unconditional branches of the successors are going to get
      // removed if we hoist the contents. We will only execute one successor
      // so assume the first successor's branch is representative.
      auto *succ_term = hoistable.front()->getTerminator();
      InstructionCost branch_cost =
          TTI.getInstructionCost(succ_term,
                                 TargetTransformInfo::TCK_RecipThroughput) +
          TTI.getInstructionCost(succ_term, TargetTransformInfo::TCK_Latency);

      // If all our successors branch to the same target, the conditional
      // branch is going to disappear as well, so we can add that to the cost
      // of the successor's branches in our analysis.
      auto *T = BB->getTerminator();
      if (new_succs.size() == 1) {
        branch_cost +=
            TTI.getInstructionCost(T, TargetTransformInfo::TCK_RecipThroughput);
        branch_cost +=
            TTI.getInstructionCost(T, TargetTransformInfo::TCK_Latency);

        // BOSCC will incur an additional cost on varying branches.
        if (UVR && UVR->isVarying(T)) {
          branch_cost += boscc_cost;
        }
      }

      // If the cost of executing everything is less than the cost of the
      // branches that would get removed, then it is beneficial to hoist.
      // If the costs are the same then we might as well make the CFG simpler!
      if (total_cost <= branch_cost) {
        // The Lower Switch Pass ought to guarantee we can only get branch
        // instructions here, but in case it didn't, we don't want to crash.
        if (auto *const Branch = dyn_cast<BranchInst>(T)) {
          for (auto *succ : hoistable) {
            modified |= hoistInstructions(*succ, *Branch, div_exceptions);
          }

          if (new_succs.size() == 1) {
            // We are not going to modify the CFG while we are working on it,
            // because that is very complex so we leave it to the Simplfy CFG
            // Pass which is to come after us, and will do a better job. So
            // here we can just pretend we modified it.
            single_succs[BB] = *new_succs.begin();
          }
        }
      }
    }
  }

  if (!modified) {
    return PreservedAnalyses::all();
  }
  return PreservedAnalyses::none();
}
