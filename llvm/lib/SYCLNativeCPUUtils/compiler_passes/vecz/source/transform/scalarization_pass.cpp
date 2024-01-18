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

#include "transform/scalarization_pass.h"

#include <compiler/utils/device_info.h>
#include <llvm/ADT/DepthFirstIterator.h>
#include <llvm/ADT/Statistic.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/Debug.h>
#include <multi_llvm/vector_type_helper.h>

#include "analysis/control_flow_analysis.h"
#include "analysis/divergence_analysis.h"
#include "analysis/uniform_value_analysis.h"
#include "analysis/vectorization_unit_analysis.h"
#include "debugging.h"
#include "llvm_helpers.h"
#include "memory_operations.h"
#include "transform/scalarizer.h"
#include "vectorization_unit.h"
#include "vecz/vecz_choices.h"
#include "vecz/vecz_target_info.h"

#define DEBUG_TYPE "vecz-scalarization"

using namespace vecz;
using namespace llvm;

STATISTIC(VeczScalarizeFail,
          "Number of kernels that failed to scalarize [ID#S80]");

ScalarizationPass::ScalarizationPass() {}

namespace {
bool needsScalarization(const Type &T) { return T.isVectorTy(); }

bool needsScalarization(const Instruction &I) {
  if (needsScalarization(*I.getType())) {
    return true;
  }
  for (const Use &op : I.operands()) {
    if (needsScalarization(*op->getType())) {
      return true;
    }
  }
  return false;
}

bool isValidScalableShuffle(const ShuffleVectorInst &shuffle) {
  // 3-element vectors are trouble, so scalarize them.
  if (!isPowerOf2_32(cast<VectorType>(shuffle.getType())
                         ->getElementCount()
                         .getFixedValue())) {
    return false;
  }
  if (!isPowerOf2_32(cast<VectorType>(shuffle.getOperand(0)->getType())
                         ->getElementCount()
                         .getFixedValue())) {
    return false;
  }
  return true;
}

bool shouldScalarize(Instruction *I, bool scalable) {
  // Don't scalarize loads or stores..
  if (isa<LoadInst>(I) || isa<StoreInst>(I)) {
    return false;
  }

  // We also don't scalarize element manipulations of load instructions
  if (auto *Shuffle = dyn_cast<ShuffleVectorInst>(I)) {
    if (scalable && !isValidScalableShuffle(*Shuffle)) {
      return true;
    }

    auto *SrcA = dyn_cast<Instruction>(Shuffle->getOperand(0));
    if (SrcA && !shouldScalarize(SrcA, scalable)) {
      return false;
    }
    auto *SrcB = dyn_cast<Instruction>(Shuffle->getOperand(1));
    if (SrcB && !shouldScalarize(SrcB, scalable)) {
      return false;
    }
  } else if (auto *Extract = dyn_cast<ExtractElementInst>(I)) {
    auto *SrcA = dyn_cast<Instruction>(Extract->getOperand(0));
    if (SrcA && !shouldScalarize(SrcA, scalable)) {
      return false;
    }
  }

  // We also don't scalarize masked memory operations
  if (auto *CI = dyn_cast<CallInst>(I)) {
    if (auto MaskedOp = MemOp::get(CI, MemOpAccessKind::Masked)) {
      if (MaskedOp->isMaskedMemOp()) {
        return false;
      }
    }
  }

  // Scalarize anything else
  return true;
}

/// @brief Operand Tracer struct
/// The purpose of this helper struct is to trace through the operands of any
/// given instruction, incrementing a usage counter, which we can compare to
/// the total number of uses for an instruction. If any instruction's counter
/// is equal to its total usage count, it has no uses other than ones we have
/// marked.
struct OperandTracer {
  using VisitSet = DenseSet<Instruction *>;

  UniformValueResult &UVR;
  bool scalable;
  VisitSet visited;
  SmallVector<Instruction *, 16> stack;

  OperandTracer(UniformValueResult &uvr, bool sc) : UVR(uvr), scalable(sc) {}

  void count(Instruction *I) {
    if (visited.insert(I).second) {
      stack.push_back(I);
    }
  }

  void countOperand(Value *V) {
    if (auto *I = dyn_cast<Instruction>(V)) {
      countInstruction(I);
    }
  }

  void countInstruction(Instruction *I) {
    if (scalable) {
      if (auto *const shuffle = dyn_cast<ShuffleVectorInst>(I)) {
        if (!isValidScalableShuffle(*shuffle)) {
          return;
        }
      }
    }

    if (I->getType()->isVectorTy() && UVR.isVarying(I)) {
      count(I);
    }
  }

  void countOperands(Instruction *I) {
    if (auto *Phi = dyn_cast<PHINode>(I)) {
      for (auto &use : Phi->incoming_values()) {
        countOperand(use.get());
      }
      return;
    }

    for (auto *V : I->operand_values()) {
      countOperand(V);
    }
  }

  void run() {
    while (!stack.empty()) {
      Instruction *I = stack.back();
      stack.pop_back();
      countOperands(I);
    }
  }
};

}  // namespace

PreservedAnalyses ScalarizationPass::run(llvm::Function &F,
                                         llvm::FunctionAnalysisManager &AM) {
  VectorizationUnit &VU = AM.getResult<VectorizationUnitAnalysis>(F).getVU();
  auto &Ctx = AM.getResult<VectorizationContextAnalysis>(F).getContext();
  const auto &MAMProxy = AM.getResult<ModuleAnalysisManagerFunctionProxy>(F);
  const auto *DI =
      MAMProxy.getCachedResult<compiler::utils::DeviceInfoAnalysis>(
          *F.getParent());
  const bool DoubleSupport = DI && DI->double_capabilities != 0;

  const bool FullScalarization =
      VU.choices().isEnabled(VectorizationChoices::eFullScalarization);
  bool NeedsScalarization = false;
  Scalarizer SR(F, Ctx, DoubleSupport);

  UniformValueResult &UVR = AM.getResult<UniformValueAnalysis>(F);

  // Find vector leaves that need to be scalarized.
  std::vector<Instruction *> Leaves;
  UVR.findVectorLeaves(Leaves);

  if (FullScalarization) {
    // Find varying vector values that need to be scalarized.
    for (BasicBlock *BB : depth_first(&F)) {
      for (Instruction &I : *BB) {
        if (needsScalarization(*I.getType()) && UVR.isVarying(&I)) {
          SR.setNeedsScalarization(&I);
          NeedsScalarization = true;
        }
      }
    }

    for (Instruction *Leaf : Leaves) {
      if (needsScalarization(*Leaf) && getVectorType(Leaf)) {
        SR.setNeedsScalarization(Leaf);
        NeedsScalarization = true;
      }
    }
  } else {
    // We use the tracer to identify instructions that are only used by
    // scalar instructions (i.e. ExtractElement instructions and reductions).
    //
    // Since these instructions don't necessarily use all lanes of their
    // operands, scalarization can produce dead code, which will get removed
    // by later cleanup optimizations. Reductions are generally much better
    // off scalarized.
    const bool scalable = VU.width().isScalable();

    OperandTracer tracer(UVR, scalable);
    for (Instruction *Leaf : Leaves) {
      if (needsScalarization(*Leaf) && getVectorType(Leaf)) {
        tracer.countOperands(Leaf);
      }
    }
    // Vector-to-scalar bitcasts aren't normally counted as vector leaves, but
    // in this case we void unnecessary scalarization if we do.
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (auto *B = dyn_cast<BitCastInst>(&I)) {
          if (B->getSrcTy()->isVectorTy() && !B->getDestTy()->isVectorTy() &&
              UVR.isVarying(B)) {
            tracer.countOperands(B);
          }
        }
      }
    }

    tracer.run();

    for (auto &BB : F) {
      for (auto &I : BB) {
        if (!shouldScalarize(&I, scalable)) {
          continue;
        }

        if (I.getType()->isVectorTy() && UVR.isVarying(&I) &&
            tracer.visited.count(&I) == 0) {
          SR.setNeedsScalarization(&I);
          NeedsScalarization = true;
        }
      }
    }
  }

  if (!NeedsScalarization) {
    return PreservedAnalyses::all();
  }

  if (!SR.scalarizeAll()) {
    ++VeczScalarizeFail;
    return VU.setFailed("Failed to scalarize");
  }

  PreservedAnalyses Preserved;
  Preserved.preserve<DominatorTreeAnalysis>();
  Preserved.preserve<LoopAnalysis>();
  Preserved.preserve<CFGAnalysis>();
  Preserved.preserve<DivergenceAnalysis>();
  return Preserved;
}
