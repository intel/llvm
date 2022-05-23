//==------ integer_misc.cpp - fallback implementation of a bunch of integer
// functions ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../device_imf.hpp"
#ifdef __LIBDEVICE_IMF_ENABLED__

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_brev(unsigned int x) {
  unsigned int res = 0;
  size_t bit_count = 8 * sizeof(unsigned int);
  for (size_t idx = 0; idx < bit_count; ++idx) {
    res |= x & 0x1;
    res <<= 1;
    x >>= 1;
  }
  return res;
}

DEVICE_EXTERN_C_INLINE
unsigned long int __devicelib_imf_brevll(unsigned long long int x) {
  unsigned long long int res = 0;
  size_t bit_count = 8 * sizeof(unsigned long long int);
  for (size_t idx = 0; idx < bit_count; ++idx) {
    res |= x & 0x1;
    res <<= 1;
    x >>= 1;
  }
  return res;
}

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_clz(int x) { return __clz(x); }

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_clzll(long long int x) { return __clzll(x); }

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_popc(unsigned int x) { return __popc(x); }

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_popcll(unsigned long long int x) { return __popcll(x); }

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_sad(int x, int y, unsigned int z) {
  return __abs(x - y) + z;
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_usad(unsigned int x, unsigned int y,
                                  unsigned int z) {
  long long int xll = x, yll = y;
  return static_cast<unsigned int>(__abs(xll - yll)) + z;
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_byte_perm(unsigned int x, unsigned int y,
                                       unsigned int s) {
  uint8_t buf[4] = {
      0,
  };
  for (size_t idx = 0; idx < 4; ++idx) {
    uint8_t select_idx = static_cast<uint8_t>(s & 0x00000007);
    if (select_idx < 4)
      buf[idx] = __get_bytes_by_index<unsigned int, uint8_t>(x, select_idx);
    else
      buf[idx] = __get_bytes_by_index<unsigned int, uint8_t>(y, select_idx - 3);
    s >>= 4;
  }
  return __assemble_integral_value<unsigned int, uint8_t, 4>(buf);
}

template <typename Ty> static inline int __do_imf_ffs(Ty x) {
  static_assert(std::is_integral<Ty>::value,
                "ffs can only accept integral type.");
  if (x == 0)
    return 0;
  size_t idx;
  for (idx = 0; idx < sizeof(Ty) * 8; ++idx) {
    if (0x1 == (0x1 & x))
      break;
    x >>= 1;
  }
  return idx + 1;
}

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_ffs(int x) { return __do_imf_ffs(x); }

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_ffsll(long long int x) { return __do_imf_ffs(x); }
#endif //__LIBDEVICE_IMF_ENABLED__
