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

#ifndef VECZ_ANALYSIS_INSTANTIATION_ANALYSIS_H_INCLUDED
#define VECZ_ANALYSIS_INSTANTIATION_ANALYSIS_H_INCLUDED

namespace llvm {
class Instruction;
}  // namespace llvm

namespace vecz {
class VectorizationContext;

/// @brief Determine whether the given instruction needs to be instantiated.
///
/// @param[in] CTx the vectorization context
/// @param[in] I Instruction to analyze.
///
/// @return true iff the instruction requires instantiation.
bool needsInstantiation(const VectorizationContext &Ctx, llvm::Instruction &I);
};  // namespace vecz

#endif  // VECZ_ANALYSIS_INSTANTIATION_ANALYSIS_H_INCLUDED
