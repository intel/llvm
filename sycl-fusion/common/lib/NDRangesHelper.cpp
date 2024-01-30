//==-------------------------- NDRangesHelper.cpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NDRangesHelper.h"

using namespace jit_compiler;
using namespace llvm;

namespace {
/// Helper class to obtain the maximal global size using a given order
///
/// This class obtains the maximal global size using the following order:
/// 1. A range is greater than another range if it contains more elements (see
/// linearize());
/// 2. Else if it appears more times in the input list of ranges;
/// 3. Else if it is greater in lexicographical order.
class IndicesFrequenceMap {
public:
  IndicesFrequenceMap()
      : GlobalLinearSize(0), IsHeterogeneousList(false), FreqMap() {}

  void add(const Indices &I);

  bool isHeterogeneousList() const { return IsHeterogeneousList; }
  const Indices &getMax() const;

private:
  std::size_t GlobalLinearSize;
  bool IsHeterogeneousList;
  // Sorted vector acting as a map.
  SmallVector<std::pair<Indices, size_t>> FreqMap;
};
} // namespace

void IndicesFrequenceMap::add(const Indices &I) {
  // From algorithm:
  // 1. A range is greater than another range if it contains more elements (see
  // linearize());
  std::size_t GLS = NDRange::linearize(I);
  if (FreqMap.empty()) {
    GlobalLinearSize = GLS;
    FreqMap.emplace_back(I, 1);
    return;
  }

  if (GLS < GlobalLinearSize) {
    // Do not add to map
    IsHeterogeneousList = true;
    return;
  }

  if (GLS == GlobalLinearSize) {
    // Add to map. Keep it sorted.
    auto *Iter = lower_bound(FreqMap, I,
                             [](const std::pair<Indices, size_t> &P,
                                const Indices &I) { return P.first < I; });
    if (Iter != FreqMap.end() && Iter->first == I) {
      // Already seen global size
      ++(Iter->second);
      return;
    }
    // New global size
    IsHeterogeneousList = true;
    FreqMap.insert(Iter, {I, 1});
    return;
  }
  // GLS > GlobalLinearSize
  // Clear map. This is the new maximum.
  GlobalLinearSize = GLS;
  IsHeterogeneousList = true;
  FreqMap.clear();
  FreqMap.emplace_back(I, 1);
}

const Indices &IndicesFrequenceMap::getMax() const {
  return std::max_element(FreqMap.begin(), FreqMap.end(),
                          [](const auto &LHS, const auto &RHS) {
                            const auto LHSN = LHS.second;
                            const auto RHSN = RHS.second;
                            // From algorithm:
                            // 2. Else if it appears more times in the input
                            // list of ranges;
                            if (LHSN < RHSN) {
                              return true;
                            }
                            if (LHSN > RHSN) {
                              return false;
                            }
                            // 3. Else if it is greater in lexicographical
                            // order.
                            return LHS.first < RHS.first;
                          })
      ->first;
}

Expected<FusedNDRange>
jit_compiler::FusedNDRange::get(ArrayRef<NDRange> NDRanges) {
  assert(!NDRanges.empty() && "Expecting at least one ND-range");
  constexpr auto GetPtrIfNonZero = [](const Indices &I) {
    return I == NDRange::AllZeros ? nullptr : &I;
  };

  constexpr auto Override = [](const Indices *&Current, const Indices *New) {
    if (Current) {
      return !New || *Current == *New;
    }
    Current = New;
    return true;
  };

  // The resulting number of dimensions will be the largest found
  int Dimensions = NDRanges.front().getDimensions();
  // The global size is given by the algorithm implemented by
  // IndicesFrequenceMap
  IndicesFrequenceMap GlobalSizeFreqMap;
  GlobalSizeFreqMap.add(NDRanges.front().getGlobalSize());
  // All local sizes and offsets must be the same
  const Indices *LocalSize = GetPtrIfNonZero(NDRanges.front().getLocalSize());
  const Indices *Offset = GetPtrIfNonZero(NDRanges.front().getOffset());
  bool IsHeterogeneousList = false;
  for (const NDRange &NDR : NDRanges.drop_front()) {
    // Accumulate dimension
    IsHeterogeneousList =
        IsHeterogeneousList || Dimensions != NDR.getDimensions();
    Dimensions = std::max(Dimensions, NDR.getDimensions());

    // Add global size to map
    GlobalSizeFreqMap.add(NDR.getGlobalSize());

    // Get local size or return error
    if (!Override(LocalSize, GetPtrIfNonZero(NDR.getLocalSize()))) {
      return createStringError(
          inconvertibleErrorCode(),
          "Cannot fuse kernels with different local sizes");
    }

    // Get offset or return error
    if (!Override(Offset, GetPtrIfNonZero(NDR.getOffset()))) {
      return createStringError(inconvertibleErrorCode(),
                               "Cannot fuse kernels with different offsets");
    }
  }

  NDRange Fused{Dimensions, GlobalSizeFreqMap.getMax(),
                LocalSize ? *LocalSize : NDRange::AllZeros,
                Offset ? *Offset : NDRange::AllZeros};
  IsHeterogeneousList =
      IsHeterogeneousList || GlobalSizeFreqMap.isHeterogeneousList();

  if (IsHeterogeneousList) {
    // ND-ranges requiring remapping cannot have non-zero offsets
    if (Offset && !all_of(NDRanges, [Fused](const NDRange &NDR) {
          return !requireIDRemapping(Fused, NDR) ||
                 NDR.getOffset() == NDRange::AllZeros;
        })) {
      return createStringError(inconvertibleErrorCode(),
                               "Cannot fuse kernels with different global "
                               "sizes in dimensions [2, N) "
                               "and non-zero offsets");
    }

    // Fused ND-range would not yield uniform work-group sizes.
    if (LocalSize && !Fused.hasUniformWorkGroupSizes()) {
      return createStringError(inconvertibleErrorCode(),
                               "Cannot fuse kernels whose fusion would "
                               "yield non-uniform work-group sizes");
    }

    // Work-items in the same work-group in the original ND-ranges must be in
    // the same work-group in the fused one.
    if (LocalSize && any_of(NDRanges, [Fused](const NDRange &NDR) {
          return NDR.hasSpecificLocalSize() && requireIDRemapping(Fused, NDR);
        })) {
      return createStringError(
          inconvertibleErrorCode(),
          "Cannot fuse kernels when any of the fused kernels with a specific "
          "local size has different global sizes in dimensions [2, N) or "
          "different number of dimensions");
    }
  }

  return FusedNDRange{Fused, IsHeterogeneousList, NDRanges};
}

bool jit_compiler::isHeterogeneousList(ArrayRef<NDRange> NDRanges) {
  const auto &ND = NDRanges.front();
  return any_of(NDRanges.drop_front(),
                [&ND](const auto &Other) { return ND != Other; });
}

bool jit_compiler::requireIDRemapping(const NDRange &LHS, const NDRange &RHS) {
  // No need to remap when all but the dimensions and the left-most components
  // of the global size range are equal.
  const auto &GS0 = LHS.getGlobalSize();
  const auto &GS1 = RHS.getGlobalSize();
  return LHS.getDimensions() != RHS.getDimensions() ||
         !std::equal(GS0.begin() + 1, GS0.end(), GS1.begin() + 1);
}
