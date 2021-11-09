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

static void *__devicelib_memcpy_uint8_aligned(void *dest, const void *src,
                                              size_t n) {
  if (dest == NULL || src == NULL || n == 0)
    return dest;

  uint8_t *dest_uint8 = reinterpret_cast<uint8_t *>(dest);
  const uint8_t *src_uint8 = reinterpret_cast<const uint8_t *>(src);
  for (size_t idx = 0; idx < n; ++idx) {
    dest_uint8[idx] = src_uint8[idx];
  }

  return dest;
}

static void *__devicelib_memcpy_uint32_aligned(void *dest, const void *src,
                                               size_t n) {
  if (dest == NULL || src == NULL || n == 0)
    return dest;

  uint32_t *dest_addr = reinterpret_cast<uint32_t *>(dest);
  const uint32_t *src_addr = reinterpret_cast<const uint32_t *>(src);
  while (n >= sizeof(uint32_t)) {
    *dest_addr++ = *src_addr++;
    n -= sizeof(uint32_t);
  }

  __devicelib_memcpy_uint8_aligned(dest_addr, src_addr, n);
  return dest;
}

DEVICE_EXTERN_C
void *__devicelib_memcpy(void *dest, const void *src, size_t n) {
  if (dest == NULL || src == NULL || n == 0)
    return dest;

  unsigned long dest_addr = reinterpret_cast<unsigned long>(dest);
  unsigned long src_addr = reinterpret_cast<unsigned long>(src);
  size_t dest_uint32_mod = dest_addr % alignof(uint32_t);
  size_t src_uint32_mod = src_addr % alignof(uint32_t);

  if (dest_uint32_mod != src_uint32_mod)
    return __devicelib_memcpy_uint8_aligned(dest, src, n);

  if (dest_uint32_mod == 0)
    return __devicelib_memcpy_uint32_aligned(dest, src, n);

  size_t head_ua_len = sizeof(uint32_t) - dest_uint32_mod;
  if (head_ua_len >= n)
    return __devicelib_memcpy_uint8_aligned(dest, src, n);
  else {
    __devicelib_memcpy_uint8_aligned(dest, src, head_ua_len);
    void *dest_aligned_addr = reinterpret_cast<void *>(dest_addr + head_ua_len);
    const void *src_aligned_addr =
        reinterpret_cast<const void *>(src_addr + head_ua_len);
    n -= head_ua_len;
    __devicelib_memcpy_uint32_aligned(dest_aligned_addr, src_aligned_addr, n);
  }

  return dest;
}

static void *__devicelib_memset_uint8_aligned(void *dest, int c, size_t n) {
  if (dest == NULL || n == 0)
    return dest;

  char *dest_addr = reinterpret_cast<char *>(dest);
  while (n > 0) {
    *dest_addr++ = c;
    --n;
  }
  return dest;
}

static void *__devicelib_memset_uint32_aligned(void *dest, int c, size_t n) {
  if (dest == NULL || n == 0)
    return dest;

  uint32_t *dest_addr = reinterpret_cast<uint32_t *>(dest);
  uint8_t temp = static_cast<uint8_t>(c);
  uint32_t memset_udw = 0;
  uint8_t *memset_udw_ptr = reinterpret_cast<uint8_t *>(&memset_udw);
  for (size_t idx = 0; idx < 4; ++idx)
    memset_udw_ptr[idx] = temp;

  while (n >= sizeof(uint32_t)) {
    *dest_addr++ = memset_udw;
    n -= sizeof(uint32_t);
  }

  __devicelib_memset_uint8_aligned(dest_addr, c, n);
  return dest;
}

DEVICE_EXTERN_C
void *__devicelib_memset(void *dest, int c, size_t n) {
  if (dest == NULL || n == 0)
    return dest;

  unsigned long memset_dest_addr = reinterpret_cast<unsigned long>(dest);
  size_t dest_uint32_mod = memset_dest_addr % alignof(uint32_t);
  if (dest_uint32_mod != 0) {
    size_t head_ua_len = sizeof(uint32_t) - dest_uint32_mod;
    if (head_ua_len >= n)
      return __devicelib_memset_uint8_aligned(dest, c, n);
    else {
      __devicelib_memset_uint8_aligned(dest, c, head_ua_len);
      n -= head_ua_len;
      memset_dest_addr += head_ua_len;
    }
  }

  __devicelib_memset_uint32_aligned(reinterpret_cast<void *>(memset_dest_addr),
                                    c, n);
  return dest;
}

static int __devicelib_memcmp_uint8_aligned(const void *s1, const void *s2,
                                            size_t n) {
  const uint8_t *s1_uint8_ptr = reinterpret_cast<const uint8_t *>(s1);
  const uint8_t *s2_uint8_ptr = reinterpret_cast<const uint8_t *>(s2);
  while (n > 0) {
    if (*s1_uint8_ptr == *s2_uint8_ptr) {
      ++s1_uint8_ptr;
      ++s2_uint8_ptr;
      --n;
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
  if (head_ua_len >= n)
    return __devicelib_memcmp_uint8_aligned(s1, s2, n);
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
