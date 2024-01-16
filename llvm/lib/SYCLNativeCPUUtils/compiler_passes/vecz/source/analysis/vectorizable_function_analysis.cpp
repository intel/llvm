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

#include "analysis/vectorizable_function_analysis.h"

#include <compiler/utils/builtin_info.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Debug.h>

#include "analysis/vectorization_unit_analysis.h"
#include "debugging.h"
#include "vectorization_context.h"

#define DEBUG_TYPE "vecz-function-analysis"

using namespace vecz;
using namespace llvm;

llvm::AnalysisKey VectorizableFunctionAnalysis::Key;

/// @brief Tell Vecz to go ahead and handle calls to declaration-only functions
///
/// This flag is for testing and debugging purposes and it should not be used
/// for normal code as instantiating undefined functions is not always valid.
cl::opt<bool> HandleDeclOnlyCalls(
    "vecz-handle-declaration-only-calls",
    cl::desc("Go ahead and handle calls to declaration-only functions"));

namespace {

/// @brief Determine whether the instruction can be vectorized or not.
///
/// @param[in] I Instruction to check for vectorizability.
/// @param[in] Ctx VectorizationContext for BuiltinInfo.
///
/// @return true if I can be vectorized.
bool canVectorize(const Instruction &I, const VectorizationContext &Ctx) {
  // Certain instructions just cannot appear.
  switch (I.getOpcode()) {
    default:
      break;
    case Instruction::IndirectBr:
    case Instruction::VAArg:
    case Instruction::Invoke:
    case Instruction::Resume:
    case Instruction::LandingPad:
      return false;
  }

  // User function calls.
  if (const CallInst *CI = dyn_cast<CallInst>(&I)) {
    if (const Function *Callee = CI->getCalledFunction()) {
      // We are going to assume that we can handle LLVM intrinsics for now and
      // let the later passes deal with them
      if (Callee->isIntrinsic()) {
        return true;
      }

      // All builtins should be vectorizable, in principle. "Invalid builtins"
      // correspond to user functions.
      const compiler::utils::BuiltinInfo &BI = Ctx.builtins();
      const auto Builtin = BI.analyzeBuiltin(*Callee);
      if (!Builtin.isValid()) {
        // If it is a user function missing a definition, we cannot safely
        // instantiate it. For example, what if it contains calls to
        // get_global_id internally?
        if (Callee->isDeclaration()) {
          return HandleDeclOnlyCalls;
        }
        // The same goes for functions we cannot inline, at least until we have
        // a way of determining if a function can be safely instantiated or not.
        if (Callee->hasFnAttribute(Attribute::NoInline)) {
          return false;
        }
      }
    }
  }

  return true;
}

/// @brief Determine whether the function can be vectorized or not.
///
/// @param[in] F Function to check for vectorizability.
/// @param[in] Ctx VectorizationContext for BuiltinInfo.
///
/// @return the Instruction that prevents the function from vectorizing, or
/// nullptr if the function can be vectorized.
const Value *canVectorize(const Function &F, const VectorizationContext &Ctx) {
  // Look for things that are not (yet?) supported.
  for (const BasicBlock &BB : F) {
    for (const Instruction &I : BB) {
      if (!canVectorize(I, Ctx)) {
        return &I;
      }
    }
  }
  return nullptr;
}

}  // namespace

VectorizableFunctionAnalysis::Result VectorizableFunctionAnalysis::run(
    llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
  Result res;
  auto &Ctx = AM.getResult<VectorizationContextAnalysis>(F).getContext();

  // Do not vectorize functions with the OptNone attribute
  if (F.hasFnAttribute(Attribute::OptimizeNone)) {
    res.canVectorize = false;
    return res;
  }

  res.failedAt = canVectorize(F, Ctx);
  res.canVectorize = !res.failedAt;
  return res;
}
