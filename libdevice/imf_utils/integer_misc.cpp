//==------ integer_misc.cpp - fallback implementation of a bunch of integer
// functions ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../device_imf.hpp"
#include <cstddef>
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
#endif //__LIBDEVICE_IMF_ENABLED__
