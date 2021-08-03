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

static int __devicelib_memcmp_1byte(const void *s1, const void *s2, size_t n) {
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

static int __devicelib_memcmp_4byte(const void *s1, const void *s2, size_t n) {
  const uint32_t *s1_uint32_ptr = reinterpret_cast<const uint32_t *>(s1);
  const uint32_t *s2_uint32_ptr = reinterpret_cast<const uint32_t *>(s2);
  while (n >= 4) {
    if (*s1_uint32_ptr == *s2_uint32_ptr) {
      s1_uint32_ptr++;
      s2_uint32_ptr++;
      n -= 4;
    } else {
      n = 4;
      break;
    }
  }

  const uint8_t *s1_uint8_ptr =
      reinterpret_cast<const uint8_t *>(s1_uint32_ptr);
  const uint8_t *s2_uint8_ptr =
      reinterpret_cast<const uint8_t *>(s2_uint32_ptr);

  for (size_t idx = 0; idx < n; ++idx) {
    if (s1_uint8_ptr[idx] == s2_uint8_ptr[idx])
      continue;
    else
      return s1_uint8_ptr[idx] - s2_uint8_ptr[idx];
  }

  return 0;
}

DEVICE_EXTERN_C
int __devicelib_memcmp(const void *s1, const void *s2, size_t n) {
  if (s1 == s2 || n == 0)
    return 0;

  size_t s1_4mod = reinterpret_cast<unsigned long>(s1) % 4;
  size_t s2_4mod = reinterpret_cast<unsigned long>(s2) % 4;

  if (s1_4mod != s2_4mod)
    return __devicelib_memcmp_1byte(s1, s2, n);

  if (s1_4mod == 0)
    return __devicelib_memcmp_4byte(s1, s2, n);

  size_t head_ua_len = 4 - s1_4mod;
  int head_cmp = __devicelib_memcmp_1byte(s1, s2, head_ua_len);
  if (head_cmp == 0) {
    const uint8_t *s1_aligned_ptr = reinterpret_cast<const uint8_t *>(s1);
    const uint8_t *s2_aligned_ptr = reinterpret_cast<const uint8_t *>(s2);
    s1_aligned_ptr += head_ua_len;
    s2_aligned_ptr += head_ua_len;
    return __devicelib_memcmp_4byte(s1_aligned_ptr, s2_aligned_ptr,
                                    n - head_ua_len);
  }

  return head_cmp;
}
#endif // __SPIR__
