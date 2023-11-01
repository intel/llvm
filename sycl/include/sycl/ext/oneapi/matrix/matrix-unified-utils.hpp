//===------- matrix-unified.hpp - SYCL matrix extension ----*- C++ -*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once
namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {
namespace matrix {

enum class use { a, b, accumulator };

enum class layout {
  row_major = 0,
  col_major = 1,
  ext_intel_packed = 2,
  dynamic = 3
};

namespace precision {
class tf32 {
  tf32() = delete;
};
} // namespace precision

} // namespace matrix
} // namespace experimental
} // namespace oneapi
} // namespace ext

namespace detail {
using UseToUseStringPair =
    std::pair<ext::oneapi::experimental::matrix::use, const char *>;

constexpr const char *
convertMatrixUseToString(ext::oneapi::experimental::matrix::use Use) {
  constexpr UseToUseStringPair UseToUseStringMap[] = {
      {ext::oneapi::experimental::matrix::use::a, "use::a"},
      {ext::oneapi::experimental::matrix::use::b, "use::b"},
      {ext::oneapi::experimental::matrix::use::accumulator, "use::accumulator"},
  };

  for (const auto &Item : UseToUseStringMap) {
    if (Item.first == Use)
      return Item.second;
  }
  return "";
}
} // namespace detail
} // namespace _V1
} // namespace sycl
