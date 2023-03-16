//==-------------------------- NDRangesHelper.cpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NDRangesHelper.h"

#include <map>

using namespace jit_compiler;
using namespace llvm;

///
/// Return the maximal global size using the following order:
/// 1. A range is greater than another range if it contains more elements (see
/// linearize());
/// 2. Else if it appears more times in the input list of ranges;
/// 3. Else if it is greater in lexicographical order.
static Indices getMaximalGlobalSize(ArrayRef<NDRange> NDRanges) {
  size_t NumElements{0};
  std::map<Indices, size_t> FreqMap;
  for (const auto &ND : NDRanges) {
    const auto &GS = ND.getGlobalSize();
    const auto N = NDRange::linearize(GS);
    if (N < NumElements) {
      continue;
    }
    if (N > NumElements) {
      NumElements = N;
      FreqMap.clear();
    }
    ++FreqMap[GS];
  }
  return std::max_element(FreqMap.begin(), FreqMap.end(),
                          [](const auto &LHS, const auto &RHS) {
                            const auto LHSN = LHS.second;
                            const auto RHSN = RHS.second;
                            if (LHSN < RHSN) {
                              return true;
                            }
                            if (LHSN > RHSN) {
                              return false;
                            }
                            return LHS.first < RHS.first;
                          })
      ->first;
}

static bool compatibleRanges(const NDRange &LHS, const NDRange &RHS) {
  const auto Dimensions = std::max(LHS.getDimensions(), RHS.getDimensions());
  const auto EqualIndices = [Dimensions](const Indices &LHS,
                                         const Indices &RHS) {
    return std::equal(LHS.begin(), LHS.begin() + Dimensions, RHS.begin());
  };
  return (!LHS.hasSpecificLocalSize() || !RHS.hasSpecificLocalSize() ||
          EqualIndices(LHS.getLocalSize(), RHS.getLocalSize())) &&
         EqualIndices(LHS.getOffset(), RHS.getOffset());
}

NDRange jit_compiler::combineNDRanges(ArrayRef<NDRange> NDRanges) {
  assert(isValidCombination(NDRanges) && "Invalid ND-ranges combination");
  const auto Dimensions =
      std::max_element(NDRanges.begin(), NDRanges.end(),
                       [](const auto &LHS, const auto &RHS) {
                         return LHS.getDimensions() < RHS.getDimensions();
                       })
          ->getDimensions();
  const auto GlobalSize = getMaximalGlobalSize(NDRanges);
  const auto *End = NDRanges.end();
  const auto *LocalSizeIter = findSpecifiedLocalSize(NDRanges);
  const auto &LocalSize =
      LocalSizeIter == End ? NDRange::AllZeros : LocalSizeIter->getLocalSize();
  const auto &Front = NDRanges.front();
  return {Dimensions, GlobalSize, LocalSize, Front.getOffset()};
}

bool jit_compiler::isHeterogeneousList(ArrayRef<NDRange> NDRanges) {
  const auto *FirstSpecifiedLocalSize = findSpecifiedLocalSize(NDRanges);
  const auto &ND = FirstSpecifiedLocalSize == NDRanges.end()
                       ? NDRanges.front()
                       : *FirstSpecifiedLocalSize;
  return any_of(NDRanges, [&ND](const auto &Other) { return ND != Other; });
}

bool jit_compiler::isValidCombination(llvm::ArrayRef<NDRange> NDRanges) {
  if (NDRanges.empty()) {
    return false;
  }
  const auto *FirstSpecifiedLocalSize = findSpecifiedLocalSize(NDRanges);
  const auto &ND = FirstSpecifiedLocalSize == NDRanges.end()
                       ? NDRanges.front()
                       : *FirstSpecifiedLocalSize;
  return llvm::all_of(NDRanges, [&ND](const auto &Other) {
    return compatibleRanges(ND, Other);
  });
}
