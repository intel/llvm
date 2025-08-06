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

#include "transform/passes.h"

#include <compiler/utils/mangling.h>
#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/Analysis/MemorySSA.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Transforms/AggressiveInstCombine/AggressiveInstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils.h>
#include <multi_llvm/vector_type_helper.h>

#include "analysis/control_flow_analysis.h"
#include "analysis/divergence_analysis.h"
#include "analysis/uniform_value_analysis.h"
#include "analysis/vectorization_unit_analysis.h"
#include "debugging.h"
#include "ir_cleanup.h"
#include "memory_operations.h"
#include "vectorization_unit.h"
#include "vecz/vecz_target_info.h"

#define DEBUG_TYPE "vecz"

using namespace llvm;

namespace vecz {
PreservedAnalyses DivergenceCleanupPass::run(Function &F,
                                             FunctionAnalysisManager &AM) {
  UniformValueResult &UVR = AM.getResult<UniformValueAnalysis>(F);

  for (BasicBlock &BB : F) {
    auto *TI = BB.getTerminator();
    if (BranchInst *Branch = dyn_cast<BranchInst>(TI)) {
      if (!Branch->isConditional()) {
        continue;
      }

      if (auto *const call = dyn_cast<CallInst>(Branch->getCondition())) {
        compiler::utils::Lexer L(call->getCalledFunction()->getName());
        if (L.Consume(VectorizationContext::InternalBuiltinPrefix) &&
            L.Consume("divergence_")) {
          // uniform reductions can just disappear
          auto *const newCond = call->getOperand(0);
          if (!UVR.isVarying(newCond)) {
            Branch->setCondition(newCond);
            if (call->use_empty()) {
              UVR.remove(call);
              call->eraseFromParent();
            }
          }
        }
      }
    }
  }

  return PreservedAnalyses::all();
}

////////////////////////////////////////////////////////////////////////////////

/// @brief Try to replace or remove masked memory operations that are trivially
/// not needed or can be converted to non-masked operations.
PreservedAnalyses SimplifyMaskedMemOpsPass::run(Function &F,
                                                FunctionAnalysisManager &AM) {
  auto &Ctx = AM.getResult<VectorizationContextAnalysis>(F).getContext();

  const TargetInfo &VTI = Ctx.targetInfo();
  std::vector<Instruction *> ToDelete;
  for (Function &Builtin : F.getParent()->functions()) {
    std::optional<MemOpDesc> BuiltinDesc =
        MemOpDesc::analyzeMaskedMemOp(Builtin);
    if (!BuiltinDesc) {
      continue;
    }
    for (User *U : Builtin.users()) {
      CallInst *CI = dyn_cast<CallInst>(U);
      if (!CI) {
        continue;
      }
      Function *Parent = CI->getParent()->getParent();
      if (Parent != &F) {
        continue;
      }
      auto MaskedOp = MemOp::get(CI, MemOpAccessKind::Masked);
      if (!MaskedOp || !MaskedOp->isMaskedMemOp()) {
        continue;
      }
      Value *Mask = MaskedOp->getMaskOperand();
      Constant *CMask = dyn_cast<Constant>(Mask);
      if (!CMask) {
        continue;
      }

      // Handle special constants.
      if (CMask->isZeroValue()) {
        // A null mask means no lane executes the memory operation.
        if (BuiltinDesc->isLoad()) {
          CI->replaceAllUsesWith(UndefValue::get(BuiltinDesc->getDataType()));
        }
        ToDelete.push_back(CI);
      } else if (CMask->isAllOnesValue()) {
        // An 'all ones' mask means all lane execute the memory operation.
        IRBuilder<> B(CI);
        Value *Data = MaskedOp->getDataOperand();
        Value *Ptr = MaskedOp->getPointerOperand();
        Type *DataTy = MaskedOp->getDataType();
        auto Alignment = BuiltinDesc->getAlignment();
        if (MaskedOp->isLoad()) {
          Value *Load = nullptr;
          if (DataTy->isVectorTy()) {
            // Skip this optimization for scalable vectors for now. It's
            // theoretically possible to perform but without scalable-vector
            // builtins we can't test it; leave any theoretical scalable-vector
            // maksed mem operation unoptimized.
            if (isa<ScalableVectorType>(DataTy)) {
              continue;
            }
            Load =
                VTI.createLoad(B, CI->getType(), Ptr, B.getInt64(1), Alignment);
          } else {
            Load = B.CreateAlignedLoad(CI->getType(), Ptr, Align(Alignment),
                                       /*isVolatile*/ false, CI->getName());
          }
          Load->takeName(CI);
          CI->replaceAllUsesWith(Load);
        } else {
          if (DataTy->isVectorTy()) {
            // Skip this optimization for scalable vectors for now. It's
            // theoretically possible to perform but without scalable-vector
            // builtins we can't test it; leave any theoretical scalable-vector
            // maksed mem operation unoptimized.
            if (isa<ScalableVectorType>(DataTy)) {
              continue;
            }
            VTI.createStore(B, Data, Ptr, B.getInt64(1),
                            BuiltinDesc->getAlignment());
          } else {
            B.CreateAlignedStore(Data, Ptr, Align(Alignment));
          }
        }
        ToDelete.push_back(CI);
      }
    }
  }

  // Clean up.
  while (!ToDelete.empty()) {
    Instruction *I = ToDelete.back();
    IRCleanup::deleteInstructionNow(I);
    ToDelete.pop_back();
  }

  PreservedAnalyses Preserved;
  Preserved.preserve<DominatorTreeAnalysis>();
  Preserved.preserve<LoopAnalysis>();
  Preserved.preserve<CFGAnalysis>();
  Preserved.preserve<DivergenceAnalysis>();
  return Preserved;
}

}  // namespace vecz
