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

#include "vectorization_heuristics.h"

#include <compiler/utils/cl_builtin_info.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Instructions.h>

#include <unordered_set>

#include "vectorization_context.h"

#define DEBUG_TYPE "vecz"

using namespace vecz;
using namespace llvm;

namespace {
class Heuristics {
  enum class BrClauseKind { None = 0, True, False };

 public:
  Heuristics(llvm::Function &F, VectorizationContext &Ctx, ElementCount VF,
             unsigned SimdDimIdx)
      : F(F), Ctx(Ctx), SimdWidth(VF), SimdDimIdx(SimdDimIdx) {}

  /// @brief Look through the scalar code to find patterns that indicate
  ///        we should not vectorize the kernel; e.g.:
  ///        __kernel Type FuncName(Params) {
  ///          if (get_global_id(0) == 0) {
  ///            // Do something.
  ///          }
  ///          // Do nothing.
  ///        }
  /// @return Whether we should vectorize the function or not.
  bool shouldVectorize();

 private:
  /// @brief Passthrough to CmpInst.
  ///
  /// @param[in] Comp The instruction to inspect.
  ///
  /// @return The branch's path not to vectorize, if any.
  BrClauseKind shouldVectorizeVisitBr(const llvm::Value *Comp) const;
  /// @brief Visit a Cmp to check if it involves a call to an opencl builtin.
  ///
  /// @param[in] Cmp The comparison instruction to inspect.
  ///
  /// @return The branch's path not to vectorize, if any.
  BrClauseKind shouldVectorizeVisitCmp(const llvm::CmpInst *Cmp) const;
  /// @brief Visit the operand of a Cmp to strip it down to a
  ///        CallInst or ConstantInt, if possible.
  ///
  /// @param[in] Val The instruction to inspect.
  /// @param[in] Cmp The comparison instruction Val belongs to.
  /// @param[in] Cache A map containing previously generated results.
  ///
  /// @return A CallInst or ConstantInt, nullptr otherwise.
  const llvm::Value *shouldVectorizeVisitCmpOperand(
      const llvm::Value *Val, const llvm::CmpInst *Cmp,
      DenseMap<const Value *, const Value *> &Cache) const;
  /// @brief Inspect the predicate and the operand that is compared against an
  ///        opencl builtin to determine if it's better not to vectorize the
  ///        kernel.
  ///
  /// @param[in] RHS  The operand compared against an opencl builtin.
  /// @param[in] Pred The kind of comparison.
  ///
  /// @return The branch's path not to vectorize, if any.
  BrClauseKind shouldVectorizeVisitCmpOperands(
      const llvm::Value *RHS, llvm::CmpInst::Predicate Pred) const;

  /// @brief The function to analyze.
  llvm::Function &F;

  /// @brief The vectorization context.
  VectorizationContext &Ctx;

  /// @brief Vectorization factor to use.
  ElementCount SimdWidth;

  /// @brief Vectorization dimension to use.
  unsigned SimdDimIdx;
};

Heuristics::BrClauseKind Heuristics::shouldVectorizeVisitCmpOperands(
    const Value *RHS, CmpInst::Predicate Pred) const {
  // If we have an `EQ` comparison, the single lane computation happens on
  // the true successor.
  if (Pred == CmpInst::Predicate::ICMP_EQ) {
    return BrClauseKind::True;
  }

  // If we have an `NE` comparison, the single lane computation happens on
  // the false successor.
  if (Pred == CmpInst::Predicate::ICMP_NE) {
    return BrClauseKind::False;
  }

  if (!RHS) {
    return BrClauseKind::None;
  }

  // If the value we compare against the opencl builtin call is a constant,
  // determine if it is worth it to vectorize based on the chances to hit a
  // branch.
  if (const ConstantInt *Val = dyn_cast<const ConstantInt>(RHS)) {
    // If we have a branch whose condition only applies for at most half of the
    // simd width, it is not worth vectorizing it.
    switch (Pred) {
      default:
        break;
      // If we have a `GT` or `GE` comparison, if the constant we compare the
      // opencl builtin against is greater than half of the simd width, we will
      // not take the true branch as often as the false branch.
      case CmpInst::Predicate::ICMP_UGT:
      case CmpInst::Predicate::ICMP_UGE:
      case CmpInst::Predicate::ICMP_SGT:
      case CmpInst::Predicate::ICMP_SGE:
        if (SimdWidth.isScalable()) {
          return BrClauseKind::True;
        } else if (Val->getValue().sgt(SimdWidth.getFixedValue() / 2)) {
          return BrClauseKind::True;
        } else if (Val->getValue().slt(SimdWidth.getFixedValue() / 2)) {
          return BrClauseKind::False;
        }
        break;
      // If we have an `LT` or `LE` comparison, if the constant we compare the
      // opencl builtin against is smaller than half of the simd width, we will
      // not take the true branch as often as the false branch.
      case CmpInst::Predicate::ICMP_ULT:
      case CmpInst::Predicate::ICMP_ULE:
      case CmpInst::Predicate::ICMP_SLT:
      case CmpInst::Predicate::ICMP_SLE:
        if (SimdWidth.isScalable()) {
          return BrClauseKind::False;
        } else if (Val->getValue().slt(SimdWidth.getFixedValue() / 2)) {
          return BrClauseKind::True;
        } else if (Val->getValue().sgt(SimdWidth.getFixedValue() / 2)) {
          return BrClauseKind::False;
        }
        break;
    }
  }

  return BrClauseKind::None;
}

const Value *Heuristics::shouldVectorizeVisitCmpOperand(
    const Value *Val, const CmpInst *Cmp,
    DenseMap<const Value *, const Value *> &Cache) const {
  const auto It = Cache.find(Val);
  if (It != Cache.end()) {
    return It->second;
  }

  // If we are visiting a binary operator, inspect both its operands.
  if (const BinaryOperator *BO = dyn_cast<const BinaryOperator>(Val)) {
    const Value *LHS =
        shouldVectorizeVisitCmpOperand(BO->getOperand(0), Cmp, Cache);
    const Value *RHS =
        shouldVectorizeVisitCmpOperand(BO->getOperand(1), Cmp, Cache);

    auto &Result = Cache[Val];

    // If any of LHS and RHS are null and the comparison instruction is not
    // an equality, Val is not constant and used in a relational comparison.
    // We don't want to work with that.
    if ((!LHS || !RHS) && !Cmp->isEquality()) {
      return (Result = nullptr);
    }

    // If the operands of the BinaryOperator are a CallInst and anything else
    // we do not want to keep going. We wish to avoid such comparisons:
    // if ((get_local_id(0) & Constant) == Constant) {}
    if (dyn_cast_or_null<const CallInst>(LHS)) {
      return (Result = nullptr);
    }
    if (dyn_cast_or_null<const CallInst>(RHS)) {
      return (Result = nullptr);
    }

    // Up to this point, LHS and RHS are either ConstantInt or null.
    if (LHS) {
      return (Result = LHS);
    }
    return (Result = RHS);
  }

  // If we are visiting an unary operator, inspect its operand.
  if (const UnaryInstruction *UI = dyn_cast<const UnaryInstruction>(Val)) {
    return shouldVectorizeVisitCmpOperand(UI->getOperand(0), Cmp, Cache);
  }

  if (const CallInst *CI = dyn_cast<const CallInst>(Val)) {
    // We only care if the CallInst does involve a call to a work-item builtin.
    const compiler::utils::BuiltinInfo &BI = Ctx.builtins();
    const auto Uniformity = BI.analyzeBuiltinCall(*CI, SimdDimIdx).uniformity;
    if (Uniformity == compiler::utils::eBuiltinUniformityInstanceID ||
        Uniformity == compiler::utils::eBuiltinUniformityMaybeInstanceID) {
      return (Cache[Val] = CI);
    }
  }

  if (const ConstantInt *CI = dyn_cast<const ConstantInt>(Val)) {
    return (Cache[Val] = CI);
  }

  return (Cache[Val] = nullptr);
}

Heuristics::BrClauseKind Heuristics::shouldVectorizeVisitCmp(
    const CmpInst *Cmp) const {
  // The following two calls return either a CallInst, a ConstantInt, or
  // nullptr otherwise. If it returns a CallInst, it necessarily is a call to
  // get_{global|local}_id, because otherwise we don't care.
  DenseMap<const Value *, const Value *> Cache;
  const Value *LHS =
      shouldVectorizeVisitCmpOperand(Cmp->getOperand(0), Cmp, Cache);
  const Value *RHS =
      shouldVectorizeVisitCmpOperand(Cmp->getOperand(1), Cmp, Cache);

  const CmpInst::Predicate pred = Cmp->getPredicate();

  BrClauseKind vectorize = BrClauseKind::None;
  // The CmpInst may involve two CallInst, or it may involve only one but
  // we don't know on which side it may be.
  if (llvm::isa_and_nonnull<const CallInst>(LHS)) {
    vectorize = shouldVectorizeVisitCmpOperands(RHS, pred);
  }
  if (llvm::isa_and_nonnull<const CallInst>(RHS)) {
    const BrClauseKind RHSStatus = shouldVectorizeVisitCmpOperands(LHS, pred);
    // This should never happen but in case it does, we want to "void" the
    // result and vectorize!
    if (vectorize != BrClauseKind::None && vectorize != RHSStatus) {
      return BrClauseKind::None;
    }
    vectorize = RHSStatus;
  }
  return vectorize;
}

Heuristics::BrClauseKind Heuristics::shouldVectorizeVisitBr(
    const Value *Comp) const {
  // If we are visiting a binary operator, inspect both its operands to
  // perhaps find CmpInsts.
  // E.g.: %and = and ...
  //       br i1 %and, ...
  if (const BinaryOperator *BO = dyn_cast<const BinaryOperator>(Comp)) {
    return (static_cast<BrClauseKind>(
        static_cast<int>(shouldVectorizeVisitBr(BO->getOperand(0))) &&
        static_cast<int>(shouldVectorizeVisitBr(BO->getOperand(1)))));
  }

  if (const CmpInst *CI = dyn_cast<const CmpInst>(Comp)) {
    return shouldVectorizeVisitCmp(CI);
  }

  return BrClauseKind::None;
}

bool Heuristics::shouldVectorize() {
  BasicBlock &BB = F.getEntryBlock();

  // Weights computed by the kind of instructions.
  // For the moment, we only consider stores/loads and function calls as being
  // expensive, without looking at what function call it is
  // (except for work item calls).
  //
  // Ultimately, it feels like this check should be done at some point during
  // the vectorization process, so that we have a better overview on how bad
  // the vectorized kernel is compared to the scalar one.
  //
  // We should most likely check only for instructions that have varying
  // operands.
  auto getWeight = [this](BasicBlock &B) {
    unsigned weight = 0;
    for (Instruction &I : B) {
      if (isa<StoreInst>(&I) || isa<LoadInst>(&I)) {
        weight++;
      } else if (CallInst *CI = dyn_cast<CallInst>(&I)) {
        const compiler::utils::BuiltinInfo &BI = Ctx.builtins();
        if (Function *Callee = CI->getCalledFunction()) {
          const auto builtin = BI.analyzeBuiltin(*Callee);
          if (!(builtin.properties &
                compiler::utils::eBuiltinPropertyWorkItem)) {
            weight++;
          }
        }
      }
    }
    return weight;
  };

  // If the program is laid out such that it may not be worth to vectorize
  // based only on the comparison of the entry block, we also have to make
  // sure that the entry block does not do as many expensive work as its
  // successors, in which case it might still be worth to vectorize.
  // We want to check if the entry block does some computation and store
  // them. Basically, if the kernel looks like:
  //
  // __kernel void FuncName(Params) {
  //   // (1) Do something.
  //   // (2) Store that something.
  //   if (get_global_id(0) == 0) {
  //     // (3) Do something.
  //   }
  //   // (4) Do nothing.
  // }
  //
  // then we might still want to vectorize it because (1) might be eligible for
  // great vectorization improvements.
  // If (2) is not present in the kernel, then we will probably not want to
  // vectorize the kernel as (1) will then either be useless or only be used
  // in (3). The former implies that it will never be used and the latter
  // implies that it will be used only once per lane, so not worth vectorizing!
  const unsigned entryBlockWeight = getWeight(BB);

  Instruction *TI = BB.getTerminator();
  if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
    if (BI->isConditional()) {
      const BrClauseKind clause = shouldVectorizeVisitBr(BI->getCondition());
      unsigned succWeight = 0;
      if (clause != BrClauseKind::None) {
        BasicBlock *start = nullptr;
        BasicBlock *terminatingBlock = nullptr;
        if (clause == BrClauseKind::True) {
          start = BI->getSuccessor(0);
          terminatingBlock = BI->getSuccessor(1);
        } else {
          start = BI->getSuccessor(1);
          terminatingBlock = BI->getSuccessor(0);
        }
        assert(terminatingBlock &&
               "Failed to get terminating block of branch inst");

        std::unordered_set<BasicBlock *> visited;
        std::vector<BasicBlock *> worklist{start};
        visited.insert(start);
        while (!worklist.empty()) {
          BasicBlock *cur = worklist.back();
          worklist.pop_back();
          succWeight += getWeight(*cur);
          for (BasicBlock *succ : successors(cur)) {
            if (succ == terminatingBlock) {
              continue;
            }
            if (visited.insert(succ).second) {
              worklist.push_back(succ);
            }
          }
        }

        // We don't want to vectorize if the path that will be taken the most
        // is the exit block of the function and does nothing else but return.
        if (isa<ReturnInst>(terminatingBlock->getTerminator()) &&
            (terminatingBlock->size() == 1) &&
            // Arbitrary limit.
            (entryBlockWeight < succWeight)) {
          return false;
        }
      }
    }
  }

  return true;
}
}  // namespace

namespace vecz {
bool shouldVectorize(llvm::Function &F, VectorizationContext &Ctx,
                     ElementCount VF, unsigned SimdDimIdx) {
  Heuristics VH(F, Ctx, VF, SimdDimIdx);
  return VH.shouldVectorize();
}
}  // namespace vecz
