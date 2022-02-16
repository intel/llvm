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
  for (size_t idx = 0; idx < n; ++idx)
    dest_uint8[idx] = src_uint8[idx];

  return dest;
}

static void *__devicelib_memcpy_uint32_aligned(void *dest, const void *src,
                                               size_t n) {
  if (dest == NULL || src == NULL || n == 0)
    return dest;

  uint32_t *dest_addr = reinterpret_cast<uint32_t *>(dest);
  const uint32_t *src_addr = reinterpret_cast<const uint32_t *>(src);
  size_t tailing_bytes = n % sizeof(uint32_t);
  size_t copy_num = n >> 2;
  size_t idx;
  for (idx = 0; idx < copy_num; ++idx)
    dest_addr[idx] = src_addr[idx];

  __devicelib_memcpy_uint8_aligned(&dest_addr[idx], &src_addr[idx],
                                   tailing_bytes);
  return dest;
}

DEVICE_EXTERN_C
void *__devicelib_memcpy(void *dest, const void *src, size_t n) {
  if (dest == NULL || src == NULL || n == 0)
    return dest;

  uintptr_t dest_addr = reinterpret_cast<uintptr_t>(dest);
  uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
  size_t dest_uint32_mod = dest_addr % alignof(uint32_t);
  size_t src_uint32_mod = src_addr % alignof(uint32_t);

  if (dest_uint32_mod != src_uint32_mod)
    return __devicelib_memcpy_uint8_aligned(dest, src, n);

  if (dest_uint32_mod == 0)
    return __devicelib_memcpy_uint32_aligned(dest, src, n);

  size_t head_ua_len = sizeof(uint32_t) - dest_uint32_mod;
  if (head_ua_len >= n)
    return __devicelib_memcpy_uint8_aligned(dest, src, n);

  __devicelib_memcpy_uint8_aligned(dest, src, head_ua_len);
  void *dest_aligned_addr = reinterpret_cast<void *>(dest_addr + head_ua_len);
  const void *src_aligned_addr =
      reinterpret_cast<const void *>(src_addr + head_ua_len);
  n -= head_ua_len;
  __devicelib_memcpy_uint32_aligned(dest_aligned_addr, src_aligned_addr, n);

  return dest;
}

static void *__devicelib_memset_uint8_aligned(void *dest, int c, size_t n) {
  if (dest == NULL || n == 0)
    return dest;

  uint8_t *dest_addr = reinterpret_cast<uint8_t *>(dest);
  for (size_t idx = 0; idx < n; ++idx)
    dest_addr[idx] = static_cast<uint8_t>(c);

  return dest;
}

static void *__devicelib_memset_uint32_aligned(void *dest, int c, size_t n) {
  if (dest == NULL || n == 0)
    return dest;

  uint32_t *dest_addr = reinterpret_cast<uint32_t *>(dest);
  uint8_t temp = static_cast<uint8_t>(c);
  uint32_t memset_udw = 0;
  uint8_t *memset_udw_ptr = reinterpret_cast<uint8_t *>(&memset_udw);
  size_t idx;
  for (idx = 0; idx < 4; ++idx)
    memset_udw_ptr[idx] = temp;

  size_t tailing_bytes = n % sizeof(uint32_t);
  size_t set_num = n >> 2;
  for (idx = 0; idx < set_num; ++idx)
    dest_addr[idx] = memset_udw;

  __devicelib_memset_uint8_aligned(&dest_addr[idx], c, tailing_bytes);
  return dest;
}

DEVICE_EXTERN_C
void *__devicelib_memset(void *dest, int c, size_t n) {
  if (dest == NULL || n == 0)
    return dest;

  uintptr_t memset_dest_addr = reinterpret_cast<uintptr_t>(dest);
  size_t dest_uint32_mod = memset_dest_addr % alignof(uint32_t);
  if (dest_uint32_mod == 0)
    return __devicelib_memset_uint32_aligned(
        reinterpret_cast<void *>(memset_dest_addr), c, n);

  size_t head_ua_len = sizeof(uint32_t) - dest_uint32_mod;
  if (head_ua_len >= n)
    return __devicelib_memset_uint8_aligned(dest, c, n);

  __devicelib_memset_uint8_aligned(dest, c, head_ua_len);
  n -= head_ua_len;
  memset_dest_addr += head_ua_len;
  __devicelib_memset_uint32_aligned(reinterpret_cast<void *>(memset_dest_addr),
                                    c, n);
  return dest;
}

static int __devicelib_memcmp_uint8_aligned(const void *s1, const void *s2,
                                            size_t n) {
  if (n == 0)
    return 0;

  const uint8_t *s1_uint8_ptr = reinterpret_cast<const uint8_t *>(s1);
  const uint8_t *s2_uint8_ptr = reinterpret_cast<const uint8_t *>(s2);

  for (size_t idx = 0; idx < n; ++idx) {
    if (s1_uint8_ptr[idx] != s2_uint8_ptr[idx])
      return s1_uint8_ptr[idx] - s2_uint8_ptr[idx];
  }

  return 0;
}

static int __devicelib_memcmp_uint32_aligned(const void *s1, const void *s2,
                                             size_t n) {
  if (n == 0)
    return 0;

  const uint32_t *s1_uint32_ptr = reinterpret_cast<const uint32_t *>(s1);
  const uint32_t *s2_uint32_ptr = reinterpret_cast<const uint32_t *>(s2);

  if (n < sizeof(uint32_t))
    return __devicelib_memcmp_uint8_aligned(s1, s2, n);

  size_t tailing_bytes = n % sizeof(uint32_t);
  size_t cmp_num = n >> 2;
  size_t idx;
  for (idx = 0; idx < cmp_num; ++idx) {
    if (s1_uint32_ptr[idx] != s2_uint32_ptr[idx])
      break;
  }

  if (idx < cmp_num)
    return __devicelib_memcmp_uint8_aligned(
        &s1_uint32_ptr[idx], &s2_uint32_ptr[idx], sizeof(uint32_t));

  if (tailing_bytes == 0)
    return 0;

  return __devicelib_memcmp_uint8_aligned(
      &s1_uint32_ptr[cmp_num], &s2_uint32_ptr[cmp_num], tailing_bytes);
}

DEVICE_EXTERN_C
int __devicelib_memcmp(const void *s1, const void *s2, size_t n) {
  if (s1 == s2 || n == 0)
    return 0;

  size_t s1_uint32_mod = reinterpret_cast<uintptr_t>(s1) % alignof(uint32_t);
  size_t s2_uint32_mod = reinterpret_cast<uintptr_t>(s2) % alignof(uint32_t);

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
