//==-------- nd_range.hpp --- SYCL iteration nd_range ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/id.hpp>
#include <CL/sycl/range.hpp>
#include <stdexcept>
#include <type_traits>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

template <int dimensions = 1> class nd_range {
  range<dimensions> globalSize;
  range<dimensions> localSize;
  id<dimensions> offset;
  static_assert(dimensions >= 1 && dimensions <= 3,
                "nd_range can only be 1, 2, or 3 dimensional.");

public:
  nd_range(range<dimensions> globalSize, range<dimensions> localSize,
           id<dimensions> offset = id<dimensions>())
      : globalSize(globalSize), localSize(localSize), offset(offset) {}

  range<dimensions> get_global_range() const { return globalSize; }

  range<dimensions> get_local_range() const { return localSize; }

  range<dimensions> get_group_range() const { return globalSize / localSize; }

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

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
