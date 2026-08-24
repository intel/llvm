//==------------------- ulp_utils.hpp - ULP comparison utility ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides checkEqual<T>(a, b, maxUlps) for floating-point types.
// Returns true if the ULP distance between a and b is <= maxUlps.
// Supports float, double, and sycl::half.
// maxUlps defaults to 0 (exact equality).
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <cstring>

// Returns true if the ULP distance between a and b is <= maxUlps.
// Uses the standard sign-magnitude to ordered-integer mapping so that
// the distance is correct across the positive/negative boundary.
template <typename T> bool checkEqual(T a, T b, unsigned maxUlps = 0) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                    std::is_same_v<T, sycl::half>,
                "checkEqual only supports float, double, or sycl::half");
  using U = std::conditional_t<
      sizeof(T) == 2, uint16_t,
      std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>>;
  // IEEE 754: -0 == +0, so handle that and the identical case together.
  if (a == b) {
    return true;
  }
  U ia, ib;
  std::memcpy(&ia, &a, sizeof(U));
  std::memcpy(&ib, &b, sizeof(U));
  U oa = (ia >> (sizeof(U) * 8 - 1)) ? ~ia : (ia | (~U(0) >> 1) + 1);
  U ob = (ib >> (sizeof(U) * 8 - 1)) ? ~ib : (ib | (~U(0) >> 1) + 1);
  return (oa > ob ? oa - ob : ob - oa) <= maxUlps;
}
