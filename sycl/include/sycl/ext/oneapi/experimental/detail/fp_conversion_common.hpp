//==------- fp_conversion_common.hpp - shared FP conversion helpers -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Definitions shared by the narrow floating-point conversion extensions
// (sycl_ext_oneapi_fp8, sycl_ext_oneapi_fp4). Keep this header free of
// format-specific logic so both extensions can include it without pulling in
// unrelated declarations.

#pragma once

#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

// Superset of the rounding modes used across the FP conversion extensions.
// An individual extension may accept only a subset (e.g. fp4 supports only
// `to_even`); the extra enumerators are harmless when unused.
enum class rounding {
  to_even,
  upward,
  toward_zero,
};

struct stochastic_seed {
  explicit stochastic_seed(uint32_t *pseed) : pseed(pseed) {}
  uint32_t *const pseed;
};

namespace detail {

// Number of bits required to represent x (i.e. floor(log2(x)) + 1).
// Returns 0 for x == 0.
template <typename T> static inline int BitWidth(T x) noexcept {
  if (x == 0u)
    return 0;
  const uint64_t v = static_cast<uint64_t>(x);
#if defined(__GNUC__) || defined(__clang__)
  return 64 - __builtin_clzll(v);
#elif defined(_MSC_VER)
  unsigned long idx;
  _BitScanReverse64(&idx, v);
  return static_cast<int>(idx) + 1;
#else
  int width = 0;
  for (uint64_t t = v; t != 0u; t >>= 1)
    ++width;
  return width;
#endif
}

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
