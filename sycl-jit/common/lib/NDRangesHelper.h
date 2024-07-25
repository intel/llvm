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
#include "llvm/Support/Error.h"

namespace jit_compiler {
class FusedNDRange {
public:
  static llvm::Expected<FusedNDRange> get(llvm::ArrayRef<NDRange> NDRanges);

  const NDRange &getNDR() const { return NDR; }
  llvm::ArrayRef<NDRange> getNDRanges() const { return NDRanges; }
  bool isHeterogeneousList() const { return IsHeterogeneousList; }

private:
  FusedNDRange() = default;
  FusedNDRange(const NDRange &NDR, bool IsHeterogeneousList,
               llvm::ArrayRef<NDRange> NDRanges)
      : NDR(NDR), IsHeterogeneousList(IsHeterogeneousList), NDRanges(NDRanges) {
  }

  NDRange NDR;
  bool IsHeterogeneousList;
  llvm::ArrayRef<NDRange> NDRanges;
};

///
/// Returns whether the input list of ND-ranges is heterogeneous or not.
bool isHeterogeneousList(llvm::ArrayRef<NDRange> NDRanges);

///
/// Return whether ID remapping will be needed.
///
/// ID remapping is needed when the number of dimensions are different or the
/// [2, N) components of the global sizes are not equal.
bool requireIDRemapping(const NDRange &LHS, const NDRange &RHS);
} // namespace jit_compiler

#endif // SYCL_FUSION_COMMON_NDRANGESHELPER_H
