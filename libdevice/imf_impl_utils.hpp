//==------- imf_impl_utils.hpp - utils definitions used by half and bfloat16 imf
// functions -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==------------------------------------------------------------------------==//

#ifndef __LIBDEVICE_IMF_IMPL_UTILS_H__
#define __LIBDEVICE_IMF_IMPL_UTILS_H__
#include <cstddef>
#include <cstdint>
// Rounding mode are used internally by type convert functions in imf libdevice
//  and we don't want to include system's fenv.h, so we define ourselves'.
typedef enum {
  __IML_RTE, // round to nearest-even
  __IML_RTZ, // round to zero
  __IML_RTP, // round to +inf
  __IML_RTN, // round to -inf
} __iml_rounding_mode;

template <typename Ty> struct __iml_get_unsigned {};
template <> struct __iml_get_unsigned<short> {
  using utype = uint16_t;
};

template <> struct __iml_get_unsigned<int> {
  using utype = uint32_t;
};

template <> struct __iml_get_unsigned<long long> {
  using utype = uint64_t;
};

// pre assumes input value is not 0.
template <typename Ty> static size_t get_msb_pos(Ty x) {
  size_t idx = 0;
  Ty mask = ((Ty)1 << (sizeof(Ty) * 8 - 1));
  for (idx = 0; idx < (sizeof(Ty) * 8); ++idx) {
    if ((x & mask) == mask)
      break;
    mask >>= 1;
  }

  return (sizeof(Ty) * 8 - 1 - idx);
}

#endif // __LIBDEVICE_IMF_IMPL_UTILS_H__
