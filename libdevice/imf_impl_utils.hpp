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

template <typename Ty> struct __iml_get_double_size_unsigned {};
template <> struct __iml_get_double_size_unsigned<uint16_t> {
  using utype = uint32_t;
};

template <> struct __iml_get_double_size_unsigned<uint32_t> {
  using utype = uint64_t;
};

/* template <> struct __iml_get_double_size_unsigned<uint64_t> {
  using utype = uint64_t;
};*/

template <typename Ty> struct __iml_fp_config {};

template <> struct __iml_fp_config<float> {
  // signed/unsigned integral type with same size
  using utype = uint32_t;
  using stype = int32_t;
  const static int32_t bias = 127;
  const static uint32_t exp_mask = 0xFF;
  const static uint32_t fra_mask = 0x7FFFFF;
  const static uint32_t nan_bits = 0x7FC00000;
  const static uint32_t pos_inf_bits = 0x7F800000;
  const static uint32_t neg_inf_bits = 0xFF800000;
  const static uint32_t max_fin_bits = 0x7F7FFFFF;
  const static uint32_t min_fin_bits = 0xFF7FFFFF;
};

template <> struct __iml_fp_config<double> {
  using utype = uint64_t;
  using stype = int64_t;
  const static int32_t bias = 1023;
  const static uint64_t exp_mask = 0x7FF;
  const static uint64_t fra_mask = 0xFFFFFFFFFFFFF;
  const static uint64_t nan_bits = 0x7FF8000000000000;
  const static uint64_t pos_inf_bits = 0x7FF0000000000000;
  const static uint64_t neg_inf_bits = 0xFFF0000000000000;
  const static uint64_t max_fin_bits = 0x7FEFFFFFFFFFFFFF;
  const static uint64_t min_fin_bits = 0xFFEFFFFFFFFFFFFF;
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

// Pre-assumption, fra is not all zero bit from bit pos idx - 1 to 0
template <typename Ty> static int get_leading_zeros_from(Ty fra, int idx) {
  Ty y = static_cast<Ty>(0x1) << (idx - 1);
  for (size_t i = 0; i < idx; ++i) {
    if ((fra & y) == y)
      return i;
    y >>= 1;
  }

  // FATAL error;
  return -1;
}

#endif // __LIBDEVICE_IMF_IMPL_UTILS_H__
