//==----------- range_rounding.hpp --- SYCL range rounding utils -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/device.hpp>

#include <tuple>

#include <stddef.h> // for size_t

namespace sycl {
inline namespace _V1 {

namespace detail {

void GetRangeRoundingSettings(size_t &MinFactor, size_t &GoodFactor,
                              size_t &MinRange);

std::tuple<std::array<size_t, 3>, bool>
getMaxWorkGroups_v2(const device &Device);

bool DisableRangeRounding();

bool RangeRoundingTrace();

template <int Dims>
std::tuple<range<Dims>, bool> getRoundedRange(range<Dims> UserRange,
                                              const device &Device) {
  range<Dims> RoundedRange = UserRange;
  // Disable the rounding-up optimizations under these conditions:
  // 1. The env var SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING is set.
  // 2. The kernel is provided via an interoperability method (this uses a
  // different code path).
  // 3. The range is already a multiple of the rounding factor.
  //
  // Cases 2 and 3 could be supported with extra effort.
  // As an optimization for the common case it is an
  // implementation choice to not support those scenarios.
  // Note that "this_item" is a free function, i.e. not tied to any
  // specific id or item. When concurrent parallel_fors are executing
  // on a device it is difficult to tell which parallel_for the call is
  // being made from. One could replicate portions of the
  // call-graph to make this_item calls kernel-specific but this is
  // not considered worthwhile.

  // Perform range rounding if rounding-up is enabled.
  if (DisableRangeRounding())
    return {range<Dims>{}, false};

  // Range should be a multiple of this for reasonable performance.
  size_t MinFactorX = 16;
  // Range should be a multiple of this for improved performance.
  size_t GoodFactor = 32;
  // Range should be at least this to make rounding worthwhile.
  size_t MinRangeX = 1024;

  // Check if rounding parameters have been set through environment:
  // SYCL_PARALLEL_FOR_RANGE_ROUNDING_PARAMS=MinRound:PreferredRound:MinRange
  GetRangeRoundingSettings(MinFactorX, GoodFactor, MinRangeX);

  // In SYCL, each dimension of a global range size is specified by
  // a size_t, which can be up to 64 bits.  All backends should be
  // able to accept a kernel launch with a 32-bit global range size
  // (i.e. do not throw an error).  The OpenCL CPU backend will
  // accept every 64-bit global range, but the GPU backends will not
  // generally accept every 64-bit global range.  So, when we get a
  // non-32-bit global range, we wrap the old kernel in a new kernel
  // that has each work item peform multiple invocations the old
  // kernel in a 32-bit global range.
  id<Dims> MaxNWGs = [&] {
    auto [MaxWGs, HasMaxWGs] = getMaxWorkGroups_v2(Device);
    if (!HasMaxWGs) {
      id<Dims> Default;
      for (int i = 0; i < Dims; ++i)
        Default[i] = (std::numeric_limits<int32_t>::max)();
      return Default;
    }

    id<Dims> IdResult;
    size_t Limit = (std::numeric_limits<int>::max)();
    for (int i = 0; i < Dims; ++i)
      IdResult[i] = (std::min)(Limit, MaxWGs[Dims - i - 1]);
    return IdResult;
  }();
  auto M = (std::numeric_limits<uint32_t>::max)();
  range<Dims> MaxRange;
  for (int i = 0; i < Dims; ++i) {
    auto DesiredSize = MaxNWGs[i] * GoodFactor;
    MaxRange[i] =
        DesiredSize <= M ? DesiredSize : (M / GoodFactor) * GoodFactor;
  }

  bool DidAdjust = false;
  auto Adjust = [&](int Dim, size_t Value) {
    if (RangeRoundingTrace())
      std::cout << "parallel_for range adjusted at dim " << Dim << " from "
                << RoundedRange[Dim] << " to " << Value << std::endl;
    RoundedRange[Dim] = Value;
    DidAdjust = true;
  };

#ifdef __SYCL_EXP_PARALLEL_FOR_RANGE_ROUNDING__
  size_t GoodExpFactor = 1;
  switch (Dims) {
  case 1:
    GoodExpFactor = 32; // Make global range multiple of {32}
    break;
  case 2:
    GoodExpFactor = 16; // Make global range multiple of {16, 16}
    break;
  case 3:
    GoodExpFactor = 8; // Make global range multiple of {8, 8, 8}
    break;
  }

  // Check if rounding parameters have been set through environment:
  // SYCL_PARALLEL_FOR_RANGE_ROUNDING_PARAMS=MinRound:PreferredRound:MinRange
  GetRangeRoundingSettings(MinFactorX, GoodExpFactor, MinRangeX);

  for (auto i = 0; i < Dims; ++i)
    if (UserRange[i] % GoodExpFactor) {
      Adjust(i, ((UserRange[i] / GoodExpFactor) + 1) * GoodExpFactor);
    }
#else
  // Perform range rounding if there are sufficient work-items to
  // need rounding and the user-specified range is not a multiple of
  // a "good" value.
  if (RoundedRange[0] % MinFactorX != 0 && RoundedRange[0] >= MinRangeX) {
    // It is sufficient to round up just the first dimension.
    // Multiplying the rounded-up value of the first dimension
    // by the values of the remaining dimensions (if any)
    // will yield a rounded-up value for the total range.
    Adjust(0, ((RoundedRange[0] + GoodFactor - 1) / GoodFactor) * GoodFactor);
  }
#endif // __SYCL_EXP_PARALLEL_FOR_RANGE_ROUNDING__
#ifdef __SYCL_FORCE_PARALLEL_FOR_RANGE_ROUNDING__
  // If we are forcing range rounding kernels to be used, we always want the
  // rounded range kernel to be generated, even if rounding isn't needed
  DidAdjust = true;
#endif // __SYCL_FORCE_PARALLEL_FOR_RANGE_ROUNDING__

  for (int i = 0; i < Dims; ++i)
    if (RoundedRange[i] > MaxRange[i])
      Adjust(i, MaxRange[i]);

  if (!DidAdjust)
    return {range<Dims>{}, false};
  return {RoundedRange, true};
}

} // namespace detail
} // namespace _V1
} // namespace sycl