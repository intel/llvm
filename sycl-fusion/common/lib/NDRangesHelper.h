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

#include <algorithm>

#include "llvm/ADT/ArrayRef.h"

namespace jit_compiler {
///
/// Combine a list of ND-ranges, obtaining the resulting "fused" ND-range.
NDRange combineNDRanges(llvm::ArrayRef<NDRange> NDRanges);

///
/// Returns whether the input list of ND-ranges is heterogeneous or not.
inline bool isHeterogeneousList(llvm::ArrayRef<NDRange> NDRanges) {
  const auto *Begin = NDRanges.begin();
  const auto *End = NDRanges.end();
  const auto *FirstSpecLocal = NDRange::findSpecifiedLocalSize(Begin, End);
  return std::any_of(Begin, End,
                     [&ND = FirstSpecLocal == End ? *Begin : *FirstSpecLocal](
                         const auto &Other) { return ND != Other; });
}
} // namespace jit_compiler

#endif // SYCL_FUSION_COMMON_NDRANGESHELPER_H
