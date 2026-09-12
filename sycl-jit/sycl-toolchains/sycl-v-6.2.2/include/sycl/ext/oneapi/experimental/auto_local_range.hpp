//==--- auto_local_range.hpp --- SYCL extension for auto range -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/range.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

template <int Dimensions> static inline range<Dimensions> auto_range() {
  static_assert(1 <= Dimensions && Dimensions <= 3);
  if constexpr (Dimensions == 3) {
    return range<Dimensions>(0, 0, 0);
  } else if constexpr (Dimensions == 2) {
    return range<Dimensions>(0, 0);
  } else {
    return range<Dimensions>(0);
  }
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
