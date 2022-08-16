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
#include <sycl/half_type.hpp>
#include <type_traits>

// _iml_half_internal is internal representation for fp16 type used in intel
// math device library. The definition here should align with definition in
// https://github.com/intel/llvm/blob/sycl/libdevice/imf_half.hpp
#if defined(__SPIR__)
typedef _Float16 _iml_half_internal;
#else
typedef uint16_t _iml_half_internal;
#endif

extern "C" {
float __imf_saturatef(float);
float __imf_copysignf(float, float);
double __imf_copysign(double, double);
_iml_half_internal __imf_copysignf16(_iml_half_internal, _iml_half_internal);
};

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace intel {
namespace math {

#if __cplusplus >= 201703L
template <typename Tp> Tp saturate(Tp x) {
  static_assert(std::is_same<Tp, float>::value,
                "sycl::ext::intel::math::saturate only supports fp32 version.");
  if constexpr (std::is_same<Tp, float>::value)
    return __imf_saturatef(x);
}

template <typename Tp> Tp copysign(Tp x, Tp y) {
  static_assert(std::is_same<Tp, float>::value ||
                    std::is_same<Tp, double>::value ||
                    std::is_same<Tp, sycl::half>::value,
                "sycl::ext::intel::math::copysign only supports fp16, fp32, "
                "fp64 version.");
  if constexpr (std::is_same<Tp, float>::value)
    return __imf_copysignf(x, y);
  if constexpr (std::is_same<Tp, double>::value)
    return __imf_copysign(x, y);
  if constexpr (std::is_same<Tp, sycl::half>::value) {
    static_assert(sizeof(sycl::half) == sizeof(_iml_half_internal),
                  "sycl::half is not compatible with _iml_half_internal.");
    _iml_half_internal xi = __builtin_bit_cast(_iml_half_internal, x);
    _iml_half_internal yi = __builtin_bit_cast(_iml_half_internal, y);
    return __builtin_bit_cast(sycl::half, __imf_copysignf16(xi, yi));
  }
}

#endif
} // namespace math
} // namespace intel
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
