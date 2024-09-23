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
template <typename Ty> size_t get_msb_pos(const Ty &x) {
  size_t idx = 0;
  Ty mask = ((Ty)1 << (sizeof(Ty) * 8 - 1));
  for (idx = 0; idx < (sizeof(Ty) * 8); ++idx) {
    if ((x & mask) == mask)
      break;
    mask >>= 1;
  }

  return (sizeof(Ty) * 8 - 1 - idx);
}

class __iml_ui128 {
public:
  __iml_ui128() = default;
  __iml_ui128(const __iml_ui128 &) = default;
  explicit __iml_ui128(uint64_t x) {
    bits[0] = x;
    bits[1] = 0;
  }

  __iml_ui128 &operator=(const __iml_ui128 &x) {
    if (this != &x) {
      this->bits[0] = x.bits[0];
      this->bits[1] = x.bits[1];
    }

    return *this;
  }

  __iml_ui128 &operator=(const uint64_t &x) {
    this->bits[0] = x;
    this->bits[1] = 0;
    return *this;
  }

  explicit operator uint64_t() { return bits[0]; }
  explicit operator uint32_t() { return static_cast<uint32_t>(bits[0]); }

  __iml_ui128 operator<<(size_t n) {
    if (n == 0)
      return *this;

    if (n >= 128)
      return static_cast<__iml_ui128>(0x0U);

    __iml_ui128 x = *this;
    if (n >= 64) {
      x.bits[1] = x.bits[0] << (n - 64);
      x.bits[0] = 0x0;
    } else {
      x.bits[1] = x.bits[1] << n;
      x.bits[1] =
          x.bits[1] |
          ((x.bits[0] & ~((static_cast<uint64_t>(0x1) << (64 - n)) - 1)) >>
           (64 - n));
      x.bits[0] = x.bits[0] << n;
    }

    return x;
  }

  __iml_ui128 operator>>(size_t n) {
    if (n == 0)
      return *this;

    if (n >= 128)
      return static_cast<__iml_ui128>(0x0U);

    __iml_ui128 x = *this;
    if (n >= 64) {
      x.bits[0] = x.bits[1] >> (n - 64);
      x.bits[1] = 0x0;
    } else {
      x.bits[0] = x.bits[0] >> n;
      x.bits[0] =
          x.bits[0] |
          ((x.bits[1] & ((static_cast<uint64_t>(0x1) << n) - 1)) << (64 - n));
      x.bits[1] = x.bits[1] >> n;
    }

    return x;
  }

  __iml_ui128 operator+(const __iml_ui128 &x) {
    __iml_ui128 res;
    res.bits[0] = this->bits[0] + x.bits[0];
    res.bits[1] = this->bits[1] + x.bits[1];
    if (res.bits[0] < this->bits[0] || res.bits[0] < x.bits[0])
      res.bits[1] += 1;
    return res;
  }

  __iml_ui128 operator+(const uint64_t &x) {
    __iml_ui128 res;
    res.bits[0] = this->bits[0] + x;
    res.bits[1] = this->bits[1];
    if (res.bits[0] < this->bits[0] || res.bits[0] < x)
      res.bits[1] += 1;
    return res;
  }

  __iml_ui128 operator+=(const __iml_ui128 &x) {
    uint64_t temp = this->bits[0] + x.bits[0];
    this->bits[1] += x.bits[1];
    if ((temp < this->bits[0]) || (temp < x.bits[0]))
      this->bits[1] += 1;
    this->bits[0] = temp;
    return *this;
  }

  __iml_ui128 operator+(int x) {
    return this->operator+(static_cast<uint64_t>(x));
  }

  __iml_ui128 operator-(const uint64_t &x) {
    __iml_ui128 res;
    res.bits[0] = this->bits[0] - x;
    res.bits[1] = this->bits[1];
    if (res.bits[0] > this->bits[0])
      res.bits[1]--;
    return res;
  }

  __iml_ui128 operator-(int x) {
    return this->operator-(static_cast<uint64_t>(x));
  }

  __iml_ui128 operator-(const __iml_ui128 &x) {
    __iml_ui128 res;
    res.bits[0] = this->bits[0] - x.bits[0];
    res.bits[1] = this->bits[1];
    if (res.bits[0] > this->bits[0])
      res.bits[1]--;
    return res;
  }

  __iml_ui128 operator-=(const __iml_ui128 &x) {
    uint64_t temp = this->bits[0];
    this->bits[0] -= x.bits[0];
    if (this->bits[0] > temp)
      this->bits[1] -= 1;
    this->bits[1] -= x.bits[1];
    return *this;
  }

  __iml_ui128 operator-=(uint64_t x) {
    uint64_t temp = this->bits[0];
    this->bits[0] -= x;
    if (this->bits[0] > temp)
      this->bits[1] -= 1;
    return *this;
  }

  __iml_ui128 operator-=(int x) {
    return this->operator-=(static_cast<uint64_t>(x));
  }

  bool operator==(const __iml_ui128 &x) {
    if (this == &x)
      return true;
    return (this->bits[0] != x.bits[0]) ? false : (this->bits[1] == x.bits[1]);
  }

  bool operator!=(const __iml_ui128 &x) { return !operator==(x); }

  bool operator==(const uint64_t &x) {
    return (this->bits[1] != 0) ? false : (this->bits[0] == x);
  }

  bool operator!=(const uint64_t &x) { return !operator==(x); }

  bool operator!=(int x) { return !operator==(static_cast<uint64_t>(x)); }
  bool operator>(const __iml_ui128 &x) {
    if (this->bits[1] > x.bits[1])
      return true;
    else if (this->bits[1] < x.bits[1])
      return false;
    else
      return (this->bits[0] > x.bits[0]);
  }

  bool operator>=(const __iml_ui128 &x) {
    return operator==(x) || operator>(x);
  }

  bool operator>(const uint64_t &x) {
    if (this->bits[1] > 0)
      return true;
    return this->bits[0] > x;
  }

  __iml_ui128 operator&(const __iml_ui128 &x) {
    __iml_ui128 res;
    res.bits[0] = this->bits[0] & x.bits[0];
    res.bits[1] = this->bits[1] & x.bits[1];
    return res;
  }

  __iml_ui128 operator&(const uint64_t &x) {
    __iml_ui128 res;
    res.bits[0] = this->bits[0] & x;
    res.bits[1] = 0x0;
    return res;
  }

  __iml_ui128 operator&(const int64_t &x) {
    __iml_ui128 res;
    uint64_t ux = static_cast<uint64_t>(x);
    res.bits[0] = this->bits[0] & ux;
    res.bits[1] = 0x0;
    return res;
  }

  __iml_ui128 operator&(const uint32_t &x) {
    __iml_ui128 res;
    res.bits[0] = this->bits[0] & x;
    res.bits[1] = 0x0;
    return res;
  }

  __iml_ui128 operator&(const int32_t &x) {
    __iml_ui128 res;
    uint32_t ux = static_cast<uint32_t>(x);
    res.bits[0] = this->bits[0] & ux;
    res.bits[1] = 0x0;
    return res;
  }

  __iml_ui128 operator&=(const __iml_ui128 &x) {
    this->bits[0] &= x.bits[0];
    this->bits[1] &= x.bits[1];
    return *this;
  }

  __iml_ui128 operator|(const __iml_ui128 &x) {
    __iml_ui128 res;
    res.bits[0] = this->bits[0] | x.bits[0];
    res.bits[1] = this->bits[1] | x.bits[1];
    return res;
  }

  __iml_ui128 operator|(const uint64_t &x) {
    __iml_ui128 res;
    res.bits[0] = this->bits[0] | x;
    res.bits[1] = 0x0;
    return res;
  }

  __iml_ui128 operator|=(const __iml_ui128 &x) {
    this->bits[0] |= x.bits[0];
    this->bits[1] |= x.bits[1];
    return *this;
  }

  __iml_ui128 operator~() {
    __iml_ui128 res;
    res.bits[0] = ~this->bits[0];
    res.bits[1] = ~this->bits[1];
    return res;
  }

  size_t ui128_msb_pos() const {
    if (this->bits[1] == 0)
      return get_msb_pos<uint64_t>(this->bits[0]);
    else
      return get_msb_pos<uint64_t>(this->bits[1]) + 64;
  }

  // overflow is not considered here.
  __iml_ui128 operator*(const __iml_ui128 &x) {
    __iml_ui128 res{0x0}, tmp1, tmp2, b1;
    size_t msb1 = this->ui128_msb_pos();
    size_t msb2 = x.ui128_msb_pos();
    size_t min_msb;
    if (msb1 < msb2) {
      min_msb = msb1;
      tmp1 = x;
      tmp2 = *this;
    } else {
      min_msb = msb2;
      tmp1 = *this;
      tmp2 = x;
    }
    for (size_t idx = 0; idx <= min_msb; ++idx) {
      b1 = static_cast<__iml_ui128>(0x1);
      b1 = b1 << idx;
      __iml_ui128 t3 = tmp2 & b1;
      if (t3 == b1) {
        res = res + tmp1;
      }
      tmp1 = tmp1 << 1;
    }
    return res;
  }

  uint64_t bits[2];
};

template <typename Ty> struct __iml_get_double_size_unsigned {};
template <> struct __iml_get_double_size_unsigned<uint16_t> {
  using utype = uint32_t;
};

template <> struct __iml_get_double_size_unsigned<uint32_t> {
  using utype = uint64_t;
};

template <> struct __iml_get_double_size_unsigned<uint64_t> {
  using utype = __iml_ui128;
};

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

static int get_leading_zeros_from(__iml_ui128 fra, int idx) {
  if (idx <= 64)
    return get_leading_zeros_from<uint64_t>(fra.bits[0], idx);
  else if (idx <= 128) {
    size_t tmp_idx = idx - 64;
    uint64_t b1(1);
    if (fra.bits[1] & ((b1 << tmp_idx) - 1)) {
      return get_leading_zeros_from<uint64_t>(fra.bits[1], tmp_idx);
    } else {
      return tmp_idx + get_leading_zeros_from<uint64_t>(fra.bits[0], 64);
    }
  } else
    return -1;
}

#endif // __LIBDEVICE_IMF_IMPL_UTILS_H__
