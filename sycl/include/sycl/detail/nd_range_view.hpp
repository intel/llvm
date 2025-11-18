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

inline namespace nd_range_view_v1 {

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
  nd_range_view(sycl::range<Dims_> &N, bool SetNumWorkGroups = false)
      : MGlobalSize(&(N[0])), MSetNumWorkGroups(SetNumWorkGroups),
        MDims{size_t(Dims_)} {}

  template <int Dims_>
  nd_range_view(sycl::range<Dims_> &GlobalSize, sycl::id<Dims_> &Offset)
      : MGlobalSize(&(GlobalSize[0])), MOffset(&(Offset[0])),
        MDims{size_t(Dims_)} {}

  template <int Dims_>
  nd_range_view(sycl::nd_range<Dims_> &ExecutionRange)
      : MGlobalSize(&(ExecutionRange.globalSize[0])),
        MLocalSize(&(ExecutionRange.localSize[0])),
        MOffset(&(ExecutionRange.offset[0])), MDims{size_t(Dims_)} {}

  sycl::detail::NDRDescT toNDRDescT() const;

  const size_t *MGlobalSize = nullptr;
  const size_t *MLocalSize = nullptr;
  const size_t *MOffset = nullptr;
  bool MSetNumWorkGroups = false;
  size_t MDims = 0;
};

} // namespace nd_range_view_v1

} // namespace detail
} // namespace _V1
} // namespace sycl
