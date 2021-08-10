//==-- fallback-cstring.cpp - fallback implementation of C string functions--=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "wrapper.h"
#include <cstdint>

#ifdef __SPIR__
DEVICE_EXTERN_C
void *__devicelib_memcpy(void *dest, const void *src, size_t n) {
  return __builtin_memcpy(dest, src, n);
}

DEVICE_EXTERN_C
void *__devicelib_memset(void *dest, int c, size_t n) {
  return __builtin_memset(dest, c, n);
}

static int __devicelib_memcmp_uint8_aligned(const void *s1, const void *s2,
                                            size_t n) {
  const uint8_t *s1_uint8_ptr = reinterpret_cast<const uint8_t *>(s1);
  const uint8_t *s2_uint8_ptr = reinterpret_cast<const uint8_t *>(s2);
  while (n > 0) {
    if (*s1_uint8_ptr == *s2_uint8_ptr) {
      s1_uint8_ptr++;
      s2_uint8_ptr++;
      n--;
    } else {
      return *s1_uint8_ptr - *s2_uint8_ptr;
    }
  }

  return 0;
}

static int __devicelib_memcmp_uint32_aligned(const void *s1, const void *s2,
                                             size_t n) {
  const uint32_t *s1_uint32_ptr = reinterpret_cast<const uint32_t *>(s1);
  const uint32_t *s2_uint32_ptr = reinterpret_cast<const uint32_t *>(s2);
  while (n >= sizeof(uint32_t)) {
    if (*s1_uint32_ptr == *s2_uint32_ptr) {
      s1_uint32_ptr++;
      s2_uint32_ptr++;
      n -= sizeof(uint32_t);
    } else {
      n = sizeof(uint32_t);
      break;
    }
  }

  return (n == 0) ? 0
                  : __devicelib_memcmp_uint8_aligned(s1_uint32_ptr,
                                                     s2_uint32_ptr, n);
}

DEVICE_EXTERN_C
int __devicelib_memcmp(const void *s1, const void *s2, size_t n) {
  if (s1 == s2 || n == 0)
    return 0;

  size_t s1_uint32_mod =
      reinterpret_cast<unsigned long>(s1) % alignof(uint32_t);
  size_t s2_uint32_mod =
      reinterpret_cast<unsigned long>(s2) % alignof(uint32_t);

  if (s1_uint32_mod != s2_uint32_mod)
    return __devicelib_memcmp_uint8_aligned(s1, s2, n);

  if (s1_uint32_mod == 0)
    return __devicelib_memcmp_uint32_aligned(s1, s2, n);

  size_t head_ua_len = sizeof(uint32_t) - s1_uint32_mod;
  int head_cmp = __devicelib_memcmp_uint8_aligned(s1, s2, head_ua_len);
  if (head_cmp == 0) {
    const uint8_t *s1_aligned_ptr = reinterpret_cast<const uint8_t *>(s1);
    const uint8_t *s2_aligned_ptr = reinterpret_cast<const uint8_t *>(s2);
    s1_aligned_ptr += head_ua_len;
    s2_aligned_ptr += head_ua_len;
    return __devicelib_memcmp_uint32_aligned(s1_aligned_ptr, s2_aligned_ptr,
                                             n - head_ua_len);
  }

  return head_cmp;
}
#endif // __SPIR__
