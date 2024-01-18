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

#include "analysis/packetization_analysis.h"

#include <compiler/utils/mangling.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Debug.h>

#include "analysis/stride_analysis.h"
#include "analysis/uniform_value_analysis.h"
#include "debugging.h"
#include "memory_operations.h"
#include "offset_info.h"
#include "vectorization_context.h"
#include "vectorization_unit.h"

#define DEBUG_TYPE "vecz"

using namespace vecz;
using namespace llvm;

namespace {
bool isDivergenceReduction(const Function &F) {
  compiler::utils::Lexer L(F.getName());
  return (L.Consume(VectorizationContext::InternalBuiltinPrefix) &&
          L.Consume("divergence_"));
}
}  // namespace

llvm::AnalysisKey PacketizationAnalysis::Key;

PacketizationAnalysisResult::PacketizationAnalysisResult(
    llvm::Function &f, StrideAnalysisResult &sar)
    : F(f), SAR(sar), UVR(sar.UVR) {
  // Vectorize branch conditions.
  for (BasicBlock &BB : F) {
    auto *TI = BB.getTerminator();
    if (UVR.isVarying(TI)) {
      markForPacketization(TI);
    }
  }

  // Then vectorize other instructions, starting at leaves.
  std::vector<Instruction *> Leaves;
  UVR.findVectorLeaves(Leaves);

  // Traverse the function from the leaves to find instructions that need to be
  // packetized.
  for (Instruction *I : Leaves) {
    markForPacketization(I);
  }
}

void PacketizationAnalysisResult::markForPacketization(Value *V) {
  if (!toPacketize.insert(V).second) {
    return;
  }

  auto *const I = dyn_cast<Instruction>(V);
  if (!I) {
    return;
  }

  if (auto *phi = dyn_cast<PHINode>(I)) {
    for (unsigned i = 0, n = phi->getNumIncomingValues(); i < n; ++i) {
      auto *const incoming = phi->getIncomingValue(i);
      if (UVR.isVarying(incoming)) {
        markForPacketization(incoming);
      }
    }
    return;
  }

  auto mo = MemOp::get(I);
  if (UVR.isMaskVarying(I)) {
    if (mo) {
      markForPacketization(mo->getMaskOperand());
      return;
    }

    if (auto *const CI = dyn_cast<CallInst>(I)) {
      Function *Callee = CI->getCalledFunction();
      if (Callee && UVR.Ctx.isInternalBuiltin(Callee) &&
          isDivergenceReduction(*Callee)) {
        markForPacketization(CI->getOperand(0));
        return;
      }
    }
  }

  if (mo) {
    auto *const ptr = mo->getPointerOperand();
    if (ptr && UVR.isVarying(ptr)) {
      const auto *info = SAR.getInfo(ptr);
      assert(info && "markForPacketization: Unable to obtain stride info");

      bool hasValidStride = info->hasStride();

      // Analyse the computed stride to see if the pointer will need to be
      // packetized. No packetization is necessary where a contiguous or
      // interleaved memop can be created, since only the pointer to the
      // first element will be used.
      if (hasValidStride) {
        // Get the pointer stride as a number of elements
        auto *const eltTy = mo->getDataType();
        if (eltTy->isVectorTy() || eltTy->isPointerTy()) {
          // No interleaved memops exist for vector element types or pointer
          // types. We can only vectorize pointer loads/stores or widen vector
          // load/stores if they are contiguous.
          const auto stride = info->getConstantMemoryStride(
              eltTy, &F.getParent()->getDataLayout());
          if (stride != 1) {
            hasValidStride = false;
          }
        } else if (!VectorType::isValidElementType(eltTy)) {
          hasValidStride = false;
        }
      }

      // Only mark the pointer for packetization if it does not have a
      // valid linear stride
      if (!hasValidStride) {
        markForPacketization(ptr);
      }
    }

    auto *const data = mo->getDataOperand();
    auto *const mask = mo->getMaskOperand();
    if (data && UVR.isVarying(data)) {
      markForPacketization(data);
    }
    if (mask && UVR.isVarying(mask)) {
      markForPacketization(mask);
    }
    return;
  }

  if (auto *const intrinsic = dyn_cast<llvm::IntrinsicInst>(I)) {
    const auto intrinsicID = intrinsic->getIntrinsicID();
    if (intrinsicID == llvm::Intrinsic::lifetime_end ||
        intrinsicID == llvm::Intrinsic::lifetime_start) {
      // We don't trace through lifetime intrinsics.
      return;
    }
  }

  // Mark any varying operands for packetization..
  for (unsigned i = 0, n = I->getNumOperands(); i != n; ++i) {
    auto *const opI = I->getOperand(i);
    if (UVR.isVarying(opI)) {
      markForPacketization(opI);
    }
  }
}

PacketizationAnalysisResult PacketizationAnalysis::run(
    Function &F, llvm::FunctionAnalysisManager &AM) {
  auto &SAR = AM.getResult<StrideAnalysis>(F);
  return Result(F, SAR);
}
