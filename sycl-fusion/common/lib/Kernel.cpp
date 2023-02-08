//==-------------------------- Kernel.cpp ----------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Kernel.h"

#include <cassert>

using namespace jit_compiler;

bool NDRange::compatibleRanges(const NDRange &LHS, const NDRange &RHS) {
  const auto Dimensions = std::max(LHS.getDimensions(), RHS.getDimensions());
  const auto EqualIndices = [Dimensions](const Indices &LHS,
                                         const Indices &RHS) {
    return std::equal(LHS.begin(), LHS.begin() + Dimensions, RHS.begin());
  };
  return (!LHS.hasSpecificLocalSize() || !RHS.hasSpecificLocalSize() ||
          EqualIndices(LHS.getLocalSize(), RHS.getLocalSize())) &&
         EqualIndices(LHS.getOffset(), RHS.getOffset());
}

NDRange::NDRange(int Dimensions, const Indices &GlobalSize,
                 const Indices &LocalSize, const Indices &Offset)
    : Dimensions{Dimensions},
      GlobalSize{GlobalSize}, LocalSize{LocalSize}, Offset{Offset} {
#ifndef NDEBUG
  const auto CheckDim = [Dimensions](const Indices &Range) {
    return std::all_of(Range.begin() + Dimensions, Range.end(),
                       [](auto D) { return D == 1; });
  };
  const auto CheckOffsetDim = [Dimensions](const Indices &Offset) {
    return std::all_of(Offset.begin() + Dimensions, Offset.end(),

                       [](auto D) { return D == 0; });
  };
#endif // NDEBUG
  assert(CheckDim(GlobalSize) &&
         "Invalid global range for number of dimensions");
  assert((CheckDim(LocalSize) || std::all_of(LocalSize.begin(), LocalSize.end(),
                                             [](auto D) { return D == 0; })) &&
         "Invalid local range for number of dimensions");
  assert(CheckOffsetDim(Offset) && "Invalid offset for number of dimensions");
}
