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

#ifndef MULTI_LLVM_TARGET_TRANSFORM_INFO_H_INCLUDED
#define MULTI_LLVM_TARGET_TRANSFORM_INFO_H_INCLUDED

#include <llvm/Analysis/TargetTransformInfo.h>
#include <multi_llvm/llvm_version.h>

namespace multi_llvm {

namespace detail {

template <typename TargetTransformInfo>
auto isLegalMaskedLoadImpl(const TargetTransformInfo &TTI, llvm::Type *Ty,
                           llvm::Align Alignment, unsigned)
    -> decltype(TTI.isLegalMaskedLoad(Ty, Alignment)) {
  return TTI.isLegalMaskedLoad(Ty, Alignment);
}

template <typename TargetTransformInfo>
auto isLegalMaskedStoreImpl(const TargetTransformInfo &TTI, llvm::Type *Ty,
                            llvm::Align Alignment, unsigned)
    -> decltype(TTI.isLegalMaskedStore(Ty, Alignment)) {
  return TTI.isLegalMaskedStore(Ty, Alignment);
}

#if LLVM_VERSION_GREATER_EQUAL(21, 0)
// TODO: Make this depend only on LLVM version once we do not have to remain
// compatible with slightly older LLVM 21 snapshots.

template <typename TargetTransformInfo>
auto isLegalMaskedLoadImpl(const TargetTransformInfo &TTI, llvm::Type *Ty,
                           llvm::Align Alignment, unsigned AddrSpace)
    -> decltype(TTI.isLegalMaskedLoad(Ty, Alignment, AddrSpace)) {
  return TTI.isLegalMaskedLoad(Ty, Alignment, AddrSpace);
}

template <typename TargetTransformInfo>
auto isLegalMaskedStoreImpl(const TargetTransformInfo &TTI, llvm::Type *Ty,
                            llvm::Align Alignment, unsigned AddrSpace)
    -> decltype(TTI.isLegalMaskedStore(Ty, Alignment, AddrSpace)) {
  return TTI.isLegalMaskedStore(Ty, Alignment, AddrSpace);
}
#endif

} // namespace detail

bool isLegalMaskedLoad(const llvm::TargetTransformInfo &TTI, llvm::Type *Ty,
                       llvm::Align Alignment, unsigned AddrSpace) {
  return detail::isLegalMaskedLoadImpl(TTI, Ty, Alignment, AddrSpace);
}

bool isLegalMaskedStore(const llvm::TargetTransformInfo &TTI, llvm::Type *Ty,
                        llvm::Align Alignment, unsigned AddrSpace) {
  return detail::isLegalMaskedStoreImpl(TTI, Ty, Alignment, AddrSpace);
}

} // namespace multi_llvm

#endif // MULTI_LLVM_TARGET_TRANSFORM_INFO_H_INCLUDED
