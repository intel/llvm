// Copyright (C) Codeplay Software Limited
//
// Licensed under the Apache License, Version 2.0 (the "License") with LLVM
// Exceptions; you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/uxlfoundation/oneapi-construction-kit/blob/main/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "analysis/vectorization_unit_analysis.h"

#define DEBUG_TYPE "vecz-unit-analysis"

using namespace vecz;

llvm::AnalysisKey VectorizationUnitAnalysis::Key;

VectorizationUnitAnalysis::Result VectorizationUnitAnalysis::run(
    llvm::Function &F, llvm::FunctionAnalysisManager &) {
  return Result{Ctx.getActiveVU(&F)};
}

#undef DEBUG_TYPE
#define DEBUG_TYPE "vecz-context-analysis"

llvm::AnalysisKey VectorizationContextAnalysis::Key;

VectorizationContextAnalysis::Result VectorizationContextAnalysis::run(
    llvm::Function &, llvm::FunctionAnalysisManager &) {
  return Result{Context};
}
