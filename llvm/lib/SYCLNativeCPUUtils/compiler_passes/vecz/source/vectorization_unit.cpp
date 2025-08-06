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

#include "vectorization_unit.h"

#include <compiler/utils/builtin_info.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/PassManagerImpl.h>
#include <llvm/Support/Debug.h>
#include <llvm/Target/TargetMachine.h>
#include <multi_llvm/multi_llvm.h>
#include <multi_llvm/vector_type_helper.h>

#include <unordered_map>
#include <unordered_set>

#include "debugging.h"
#include "vectorization_context.h"
#include "vectorization_helpers.h"
#include "vecz/vecz_choices.h"

#define DEBUG_TYPE "vecz"

using namespace vecz;
using namespace llvm;

VectorizationUnit::VectorizationUnit(Function &F, ElementCount Width,
                                     unsigned Dimension,
                                     VectorizationContext &Ctx,
                                     const VectorizationChoices &Ch)
    : Ctx(Ctx),
      Choices(Ch),
      ScalarFn(&F),
      VectorizedFn(nullptr),
      SimdWidth(ElementCount()),
      LocalSize(0),
      AutoSimdWidth(false),
      SimdDimIdx(Dimension),
      FnFlags(eFunctionNoFlag) {
  // Gather information about the function's arguments.
  for (Argument &Arg : F.args()) {
    VectorizerTargetArgument TargetArg;
    TargetArg.OldArg = &Arg;
    TargetArg.NewArg = nullptr;
    TargetArg.IsVectorized = false;
    TargetArg.PointerRetPointeeTy = nullptr;
    TargetArg.Placeholder = nullptr;
    Arguments.push_back(TargetArg);
  }

  // Set the desired SIMD width and try to look up the vectorized function.
  setWidth(Width);
}

VectorizationUnit::~VectorizationUnit() {}

Function &VectorizationUnit::function() {
  if (VectorizedFn) {
    return *VectorizedFn;
  } else {
    return *ScalarFn;
  }
}

const Function &VectorizationUnit::function() const {
  if (VectorizedFn) {
    return *VectorizedFn;
  } else {
    return *ScalarFn;
  }
}

void VectorizationUnit::setWidth(ElementCount NewWidth) {
  if (NewWidth == SimdWidth) {
    return;
  }
  SimdWidth = NewWidth;

  // Determine the vectorized function's name and try to look it up.
  const std::string VectorizedName =
      getVectorizedFunctionName(ScalarFn->getName(), SimdWidth, Choices);
  if (VectorizedFn) {
    VectorizedFn->setName(VectorizedName);
  } else {
    setVectorizedFunction(Ctx.module().getFunction(VectorizedName));
  }
}

void VectorizationUnit::setScalarFunction(llvm::Function *NewFunction) {
  if (!NewFunction) {
    return;
  }
  ScalarFn = NewFunction;
  unsigned i = 0;
  for (Argument &Arg : NewFunction->args()) {
    VectorizerTargetArgument &TargetArg = Arguments[i];
    TargetArg.OldArg = &Arg;
    i++;
  }
}

void VectorizationUnit::setVectorizedFunction(llvm::Function *NewFunction) {
  VectorizedFn = NewFunction;
  ArgumentPlaceholders.clear();
  if (!NewFunction) {
    for (unsigned i = 0; i < Arguments.size(); i++) {
      VectorizerTargetArgument &TargetArg = Arguments[i];
      TargetArg.NewArg = nullptr;
      TargetArg.Placeholder = nullptr;
    }
  } else {
    unsigned i = 0;
    for (Argument &Arg : NewFunction->args()) {
      VectorizerTargetArgument &TargetArg = Arguments[i];
      TargetArg.NewArg = &Arg;

      Instruction *Placeholder = nullptr;
      if (TargetArg.IsVectorized && !TargetArg.PointerRetPointeeTy &&
          !Arg.user_empty()) {
        // A vectorized argument will be used only by its placeholder extract
        // element instruction
        Placeholder = cast<Instruction>(*Arg.user_begin());
      }

      TargetArg.Placeholder = Placeholder;
      if (Placeholder) {
        // Mark the extract to distinguish them from other instructions.
        ArgumentPlaceholders.insert(Placeholder);
      }
      i++;
    }
  }
}

vecz::internal::AnalysisFailResult VectorizationUnit::setFailed(
    const char *remark, const llvm::Function *F, const llvm::Value *V) {
  setFlag(eFunctionVectorizationFailed);
  emitVeczRemarkMissed(F ? F : &function(), V, remark);
  return vecz::internal::AnalysisFailResult();
}

VectorizationResult VectorizationUnit::getResult() const {
  VectorizationResult res;
  res.func = VectorizedFn;

  for (const VectorizerTargetArgument &TargetArg : Arguments) {
    Type *pointerRetPointeeTy = nullptr;
    VectorizationResult::Arg::Kind kind = VectorizationResult::Arg::SCALAR;
    if (auto *ty = TargetArg.PointerRetPointeeTy) {
      pointerRetPointeeTy = ty;
      kind = VectorizationResult::Arg::POINTER_RETURN;
    } else if (TargetArg.IsVectorized) {
      kind = VectorizationResult::Arg::VECTORIZED;
    }
    res.args.emplace_back(kind, TargetArg.NewArg->getType(),
                          pointerRetPointeeTy);
  }
  return res;
}
