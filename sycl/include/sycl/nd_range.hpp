//==-------- nd_range.hpp --- SYCL iteration nd_range ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <stdexcept>
#include <sycl/id.hpp>
#include <sycl/range.hpp>
#include <type_traits>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

/// Defines the iteration domain of both the work-groups and the overall
/// dispatch.
///
/// \ingroup sycl_api
template <int dimensions = 1> class nd_range {
  range<dimensions> globalSize;
  range<dimensions> localSize;
  id<dimensions> offset;
  static_assert(dimensions >= 1 && dimensions <= 3,
                "nd_range can only be 1, 2, or 3 dimensional.");

public:
  __SYCL2020_DEPRECATED("offsets are deprecated in SYCL2020")
  nd_range(range<dimensions> globalSize, range<dimensions> localSize,
           id<dimensions> offset)
      : globalSize(globalSize), localSize(localSize), offset(offset) {}

  nd_range(range<dimensions> globalSize, range<dimensions> localSize)
      : globalSize(globalSize), localSize(localSize), offset(id<dimensions>()) {
  }

  range<dimensions> get_global_range() const { return globalSize; }

  range<dimensions> get_local_range() const { return localSize; }

  range<dimensions> get_group_range() const { return globalSize / localSize; }

  __SYCL2020_DEPRECATED("offsets are deprecated in SYCL2020")
  id<dimensions> get_offset() const { return offset; }

  // Common special member functions for by-value semantics
  nd_range(const nd_range<dimensions> &rhs) = default;
  nd_range(nd_range<dimensions> &&rhs) = default;
  nd_range<dimensions> &operator=(const nd_range<dimensions> &rhs) = default;
  nd_range<dimensions> &operator=(nd_range<dimensions> &&rhs) = default;
  nd_range() = default;

  // Common member functions for by-value semantics
  bool operator==(const nd_range<dimensions> &rhs) const {
    return (rhs.globalSize == this->globalSize) &&
           (rhs.localSize == this->localSize) && (rhs.offset == this->offset);
  }

  bool operator!=(const nd_range<dimensions> &rhs) const {
    return !(*this == rhs);
  }
};

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
