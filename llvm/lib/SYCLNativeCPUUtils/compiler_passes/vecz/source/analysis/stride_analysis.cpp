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

#include "analysis/stride_analysis.h"

#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Debug.h>

#include "analysis/uniform_value_analysis.h"
#include "debugging.h"
#include "memory_operations.h"
#include "offset_info.h"
#include "vectorization_context.h"
#include "vectorization_unit.h"

#define DEBUG_TYPE "vecz"

using namespace vecz;
using namespace llvm;

llvm::AnalysisKey StrideAnalysis::Key;

OffsetInfo &StrideAnalysisResult::analyze(Value *V) {
  const auto find = analyzed.find(V);
  if (find != analyzed.end()) {
    return find->second;
  }

  // We construct it on the stack first, and copy it into the map, because
  // the constructor itself can create more things in the map and constructing
  // it in-place could result in the storage being re-allocated while the
  // constructor is still running.
  const auto OI = OffsetInfo(*this, V);
  return analyzed.try_emplace(V, OI).first->second;
}

StrideAnalysisResult::StrideAnalysisResult(llvm::Function &f,
                                           UniformValueResult &uvr,
                                           AssumptionCache &AC)
    : F(f), UVR(uvr), AC(AC) {
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (!UVR.isVarying(&I)) {
        continue;
      }

      if (auto mo = MemOp::get(&I)) {
        auto *const ptr = mo->getPointerOperand();
        analyze(ptr);
      }
    }
  }
}

void StrideAnalysisResult::manifestAll(IRBuilder<> &B) {
  const auto saved = B.GetInsertPoint();
  for (auto &info : analyzed) {
    info.second.manifest(B, *this);
  }
  B.SetInsertPoint(saved->getParent(), saved);
}

Value *StrideAnalysisResult::buildMemoryStride(IRBuilder<> &B, llvm::Value *Ptr,
                                               llvm::Type *EleTy) const {
  if (auto *const info = getInfo(Ptr)) {
    return info->buildMemoryStride(B, EleTy, &F.getParent()->getDataLayout());
  }
  return nullptr;
}

StrideAnalysisResult StrideAnalysis::run(llvm::Function &F,
                                         llvm::FunctionAnalysisManager &AM) {
  auto &AC = AM.getResult<AssumptionAnalysis>(F);
  auto &UVR = AM.getResult<UniformValueAnalysis>(F);
  return Result(F, UVR, AC);
}

PreservedAnalyses StrideAnalysisPrinterPass::run(Function &F,
                                                 FunctionAnalysisManager &AM) {
  auto &SAR = AM.getResult<StrideAnalysis>(F);
  OS << "StrideAnalysis for function '" << F.getName() << "':\n";

  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto MO = MemOp::get(&I)) {
        auto *const Ptr = MO->getPointerOperand();
        if (!Ptr) {
          continue;
        }
        if (const OffsetInfo *Info = SAR.getInfo(Ptr)) {
          OS << "* Stride for " << *Ptr << "\n  - ";
          if (Info->mayDiverge()) {
            OS << "divergent";
          } else if (Info->hasStride()) {
            OS << "linear";
          } else if (Info->isUniform()) {
            OS << "uniform";
          } else {
            OS << "unknown";
          }
          if (Info->isStrideConstantInt()) {
            OS << " stride of " << Info->getStrideAsConstantInt();
          }
          OS << "\n";
        }
      }
    }
  }
  return PreservedAnalyses::all();
}
