//==---- nd_range_view.hpp --- SYCL iteration with reference to ranges ---==//
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
namespace detail {

class NDRDescT;

// The structure to keep dimension and references to ranges unified for
// all dimensions.
class nd_range_view {

public:
  nd_range_view() = default;
  nd_range_view(const nd_range_view &Desc) = default;
  nd_range_view(nd_range_view &&Desc) = default;
  nd_range_view &operator=(const nd_range_view &Desc) = default;
  nd_range_view &operator=(nd_range_view &&Desc) = default;

  template <int Dims_>
  nd_range_view(sycl::range<Dims_> &GlobalSizes, sycl::range<Dims_> &LocalSizes)
      : GlobalSize(&(GlobalSizes[0])), LocalSize(&(LocalSizes[0])),
        Dims{size_t(Dims_)} {}

  // to support usage in sycl::ext::oneapi::experimental::submit_with_event()
  template <int Dims_>
  nd_range_view(sycl::nd_range<Dims_> &ExecutionRange)
      : GlobalSize(&ExecutionRange.globalSize[0]),
        LocalSize(&ExecutionRange.localSize[0]),
        Offset(&ExecutionRange.offset[0]), Dims{size_t(Dims_)} {}

  template <int Dims_>
  nd_range_view(sycl::range<Dims_> &Range)
      : GlobalSize(&(Range[0])), Dims{size_t(Dims_)} {}

  sycl::detail::NDRDescT toNDRDescT() const;

  const size_t *GlobalSize = nullptr;
  const size_t *LocalSize = nullptr;
  const size_t *Offset = nullptr;
  size_t Dims = 0;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
