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

DEVICE_EXTERN_C
int __devicelib_memcmp(const void *s1, const void *s2, size_t n) {
  if (s1 == s2 || n == 0)
    return 0;

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
#endif // __SPIR__
