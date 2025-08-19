//==---- ranges_ref_view.hpp --- SYCL iteration with reference to ranges ---==//
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

// The structure to keep dimension and references to ranges unified for
// all dimensions.
class ranges_ref_view {

public:
  ranges_ref_view() = default;
  ranges_ref_view(const ranges_ref_view &Desc) = default;
  ranges_ref_view(ranges_ref_view &&Desc) = default;
  ranges_ref_view &operator=(const ranges_ref_view &Desc) = default;
  ranges_ref_view &operator=(ranges_ref_view &&Desc) = default;

  template <int Dims_>
  ranges_ref_view(sycl::range<Dims_> &GlobalSizes,
                  sycl::range<Dims_> &LocalSizes)
      : GlobalSize(&(GlobalSizes[0])), LocalSize(&(LocalSizes[0])),
        Dims{size_t(Dims_)} {}

  // to support usage in sycl::ext::oneapi::experimental::submit_with_event()
  template <int Dims_>
  ranges_ref_view(sycl::nd_range<Dims_> &ExecutionRange)
      : GlobalSize(&ExecutionRange.globalSize[0]),
        LocalSize(&ExecutionRange.localSize[0]),
        GlobalOffset(&ExecutionRange.offset[0]), Dims{size_t(Dims_)} {}

  template <int Dims_>
  ranges_ref_view(sycl::range<Dims_> &Range)
      : GlobalSize(&(Range[0])), Dims{size_t(Dims_)} {}

  const size_t *GlobalSize = nullptr;
  const size_t *LocalSize = nullptr;
  const size_t *GlobalOffset = nullptr;
  size_t Dims = 0;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
