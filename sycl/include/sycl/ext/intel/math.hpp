//==------------- math.hpp - Intel specific math API -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The main header of Intel specific math API
//===----------------------------------------------------------------------===//

#pragma once
#include <type_traits>
extern "C" {
float __imf_saturatef(float);
float __imf_copysignf(float, float);
double __imf_copysign(double, double);
};

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace intel {
namespace math {
template <typename Tp> Tp saturate(Tp x) {
  static_assert(std::is_same<Tp, float>::value,
                "sycl::ext::intel::math::saturate only supports fp32 version.");
  if (std::is_same<Tp, float>::value)
    return __imf_saturatef(x);
}

template <typename Tp> Tp copysign(Tp x, Tp y) {
  static_assert(
      std::is_same<Tp, float>::value || std::is_same<Tp, double>::value,
      "sycl::ext::intel::math::copysign only supports fp32, fp64 version.");
  if (std::is_same<Tp, float>::value)
    return __imf_copysignf(x, y);
  if (std::is_same<Tp, double>::value)
    return __imf_copysign(x, y);
}

} // namespace math
} // namespace intel
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
