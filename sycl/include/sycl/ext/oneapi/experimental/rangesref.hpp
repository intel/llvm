//==-------- rangesref.hpp --- SYCL iteration with reference to ranges -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/nd_range.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

// The structure to keep references to ranges and dimension unified for
// all dimensions.
class RangesRefT {

public:
  RangesRefT() = default;
  RangesRefT(const RangesRefT &Desc) = default;
  RangesRefT(RangesRefT &&Desc) = default;

  template <int Dims_>
  RangesRefT(sycl::range<Dims_> &GlobalSizes, sycl::range<Dims_> &LocalSizes)
      : GlobalSize(&(GlobalSizes[0])), LocalSize(&(LocalSizes[0])),
        Dims{size_t(Dims_)} {}

  // to support usage in sycl::ext::oneapi::experimental::submit_with_event()
  template <int Dims_>
  RangesRefT(sycl::nd_range<Dims_> &ExecutionRange)
      : GlobalSize(&ExecutionRange.globalSize[0]),
        LocalSize(&ExecutionRange.localSize[0]),
        GlobalOffset(&ExecutionRange.offset[0]), Dims{size_t(Dims_)} {}

  template <int Dims_>
  RangesRefT(sycl::range<Dims_> &Range)
      : GlobalSize(&(Range[0])), Dims{size_t(Dims_)} {}

  RangesRefT &operator=(const RangesRefT &Desc) = default;
  RangesRefT &operator=(RangesRefT &&Desc) = default;

  const size_t *GlobalSize = nullptr;
  const size_t *LocalSize = nullptr;
  const size_t *GlobalOffset = nullptr;
  size_t Dims = 0;
};

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
