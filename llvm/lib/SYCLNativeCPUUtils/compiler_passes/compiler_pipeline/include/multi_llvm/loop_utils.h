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
#ifndef MULTI_LLVM_LOOP_UTILS_H_INCLUDED
#define MULTI_LLVM_LOOP_UTILS_H_INCLUDED

#include <llvm/Transforms/Utils/LoopUtils.h>
#include <multi_llvm/llvm_version.h>

namespace multi_llvm {

inline llvm::Value *createSimpleTargetReduction(
    llvm::IRBuilderBase &B, const llvm::TargetTransformInfo *TTI,
    llvm::Value *Src, llvm::RecurKind RdxKind) {
#if LLVM_VERSION_MAJOR >= 20
  (void)TTI;
  return llvm::createSimpleReduction(B, Src, RdxKind);
#elif LLVM_VERSION_MAJOR >= 18
  (void)TTI;
  return llvm::createSimpleTargetReduction(B, Src, RdxKind);
#else
  return llvm::createSimpleTargetReduction(B, TTI, Src, RdxKind);
#endif
}

}  // namespace multi_llvm

#endif  // MULTI_LLVM_LOOP_UTILS_H_INCLUDED
