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

#ifndef VECZ_VECTORIZATION_HEURISTICS_H_INCLUDED
#define VECZ_VECTORIZATION_HEURISTICS_H_INCLUDED

#include <llvm/Support/TypeSize.h>

namespace llvm {
class Function;
}  // namespace llvm

namespace vecz {
class VectorizationContext;

/// @brief Decide whether a function is worth vectorizing for a given
/// vectorization factor.
///
/// @param[in] F the function to analyze
/// @param[in] Ctx the vectorization context
/// @param[in] VF the vectorization factor
/// @param[in] SimdDimIdx the vectorization dimension
///
/// @return Whether we should vectorize the function or not.
bool shouldVectorize(llvm::Function &F, VectorizationContext &Ctx,
                     llvm::ElementCount VF, unsigned SimdDimIdx);

}  // namespace vecz

#endif  // VECZ_VECTORIZATION_HEURISTICS_H_INCLUDED
