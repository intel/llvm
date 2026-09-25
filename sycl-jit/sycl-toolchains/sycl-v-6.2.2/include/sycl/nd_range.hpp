//==-------- nd_range.hpp --- SYCL iteration nd_range ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp> // for __SYCL2020_DEPRECATED
#include <sycl/id.hpp>                        // for id
#include <sycl/range.hpp>                     // for range

namespace sycl {
inline namespace _V1 {

/// Defines the iteration domain of both the work-groups and the overall
/// dispatch.
///
/// \ingroup sycl_api
template <int Dimensions = 1> class nd_range {
public:
  static constexpr int dimensions = Dimensions;

private:
  range<Dimensions> globalSize;
  range<Dimensions> localSize;
  id<Dimensions> offset;
  static_assert(Dimensions >= 1 && Dimensions <= 3,
                "nd_range can only be 1, 2, or 3 Dimensional.");

public:
  __SYCL2020_DEPRECATED("offsets are deprecated in SYCL2020")
  nd_range(range<Dimensions> globalSize, range<Dimensions> localSize,
           id<Dimensions> offset)
      : globalSize(globalSize), localSize(localSize), offset(offset) {}

  nd_range(range<Dimensions> globalSize, range<Dimensions> localSize)
      : globalSize(globalSize), localSize(localSize), offset(id<Dimensions>()) {
  }

  range<Dimensions> get_global_range() const { return globalSize; }

  range<Dimensions> get_local_range() const { return localSize; }

  range<Dimensions> get_group_range() const { return globalSize / localSize; }

  __SYCL2020_DEPRECATED("offsets are deprecated in SYCL2020")
  id<Dimensions> get_offset() const { return offset; }

  // Common special member functions for by-value semantics
  nd_range(const nd_range<Dimensions> &rhs) = default;
  nd_range(nd_range<Dimensions> &&rhs) = default;
  nd_range<Dimensions> &operator=(const nd_range<Dimensions> &rhs) = default;
  nd_range<Dimensions> &operator=(nd_range<Dimensions> &&rhs) = default;
  nd_range() = default;

  // Common member functions for by-value semantics
  bool operator==(const nd_range<Dimensions> &rhs) const {
    return (rhs.globalSize == this->globalSize) &&
           (rhs.localSize == this->localSize) && (rhs.offset == this->offset);
  }

  bool operator!=(const nd_range<Dimensions> &rhs) const {
    return !(*this == rhs);
  }
};

} // namespace _V1
} // namespace sycl
