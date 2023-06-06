//==--------- NDRangesHelper.h - Helpers to handle ND-ranges ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_FUSION_COMMON_NDRANGESHELPER_H
#define SYCL_FUSION_COMMON_NDRANGESHELPER_H

#include "Kernel.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"

namespace jit_compiler {
///
/// Combine a list of ND-ranges, obtaining the resulting "fused" ND-range.
NDRange combineNDRanges(llvm::ArrayRef<NDRange> NDRanges);

inline llvm::ArrayRef<NDRange>::const_iterator
findSpecifiedLocalSize(llvm::ArrayRef<NDRange> NDRanges) {
  return llvm::find_if(
      NDRanges, [](const auto &ND) { return ND.hasSpecificLocalSize(); });
}

///
/// Returns whether the input list of ND-ranges is heterogeneous or not.
bool isHeterogeneousList(llvm::ArrayRef<NDRange> NDRanges);

///
/// Return whether a combination of ND-ranges is valid for fusion.
bool isValidCombination(llvm::ArrayRef<NDRange> NDRanges);

///
/// Return whether ID remapping will be needed.
///
/// ID remapping is needed when the number of dimensions are different or the
/// [2, N) components of the global sizes are not equal.
bool requireIDRemapping(const NDRange &LHS, const NDRange &RHS);
} // namespace jit_compiler

#endif // SYCL_FUSION_COMMON_NDRANGESHELPER_H
