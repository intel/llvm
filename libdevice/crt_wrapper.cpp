//==------ crt_wrapper.cpp - wrappers for libc internal functions ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/spir_global_var.hpp"
#include "spirv_vars.h"
#include "wrapper.h"

#include <cstdint>

#define RAND_NEXT_LEN 1024
DeviceGlobal<uint64_t[RAND_NEXT_LEN]> RandNext;

#if defined(__SPIR__) || defined(__SPIRV__) || defined(__NVPTX__) ||           \
    defined(__AMDGCN__)
DEVICE_EXTERN_C_INLINE
void *memcpy(void *dest, const void *src, size_t n) {
  return __devicelib_memcpy(dest, src, n);
}

DEVICE_EXTERN_C_INLINE
void *memset(void *dest, int c, size_t n) {
  return __devicelib_memset(dest, c, n);
}

DEVICE_EXTERN_C_INLINE
int memcmp(const void *s1, const void *s2, size_t n) {
  return __devicelib_memcmp(s1, s2, n);
}

// We align with libc string functions, no checking for null ptr.
DEVICE_EXTERN_C_INLINE
size_t strlen(const char *s) {
  const char *t;
  for (t = s; *t; ++t)
    ;
  return t - s;
}

DEVICE_EXTERN_C_INLINE
char *strcpy(char *dest, const char *src) {
  char *t = dest;
  for (; (*dest = *src) != '\0'; ++dest, ++src)
    ;
  return t;
}

DEVICE_EXTERN_C_INLINE
char *strncpy(char *dest, const char *src, size_t n) {
  size_t i;
  for (i = 0; i < n && (src[i] != '\0'); ++i)
    dest[i] = src[i];
  for (; i < n; ++i)
    dest[i] = '\0';
  return dest;
}

DEVICE_EXTERN_C_INLINE
int strcmp(const char *s1, const char *s2) {
  while (*s1 == *s2) {
    if (*s1 == '\0')
      return 0;
    ++s1;
    ++s2;
  }

  return *reinterpret_cast<const unsigned char *>(s1) -
         *reinterpret_cast<const unsigned char *>(s2);
}

DEVICE_EXTERN_C_INLINE
int strncmp(const char *s1, const char *s2, size_t n) {

  size_t idx = 0;
  while ((idx < n) && (s1[idx] != '\0') && (s1[idx] == s2[idx]))
    idx++;

  if (idx == n)
    return 0;

  return s1[idx] - s2[idx];
}

// This simple rand is for ease of use only, the implementation aligns with
// LLVM libc rand which is based on xorshift64star pseudo random number
// generator. If work item number <= 1024, each work item has its own internal
// state stored in RandNext, no data race happens and the sequence of the value
// generated can be reproduced from run to run. If work item number > 1024,
// multiple work item may share same 'RandNext' value, data race happens and
// the value generated can't be reproduced from run to run.
#define RAND_MAX 0x7fffffff
#ifdef __SYCL_DEVICE_ONLY__
#define RAND_NEXT_ACC RandNext.get()
#endif

DEVICE_EXTERN_C_INLINE
int rand() {
  size_t gid =
      (__spirv_BuiltInGlobalInvocationId(0) * __spirv_BuiltInGlobalSize(1) *
       __spirv_BuiltInGlobalSize(2)) +
      (__spirv_BuiltInGlobalInvocationId(1) * __spirv_BuiltInGlobalSize(2)) +
      __spirv_BuiltInGlobalInvocationId(2);
  size_t global_size = __spirv_BuiltInGlobalSize(0) *
                       __spirv_BuiltInGlobalSize(1) *
                       __spirv_BuiltInGlobalSize(2);
  size_t gid1 =
      (global_size > RAND_NEXT_LEN) ? (gid & (RAND_NEXT_LEN - 1)) : gid;
  if (RAND_NEXT_ACC[gid1] == 0)
    RAND_NEXT_ACC[gid1] = 1;
  uint64_t x = RAND_NEXT_ACC[gid1];
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  RAND_NEXT_ACC[gid1] = x;
  return static_cast<int>((x * 0x2545F4914F6CDD1Dul) >> 32) & RAND_MAX;
}

DEVICE_EXTERN_C_INLINE
void srand(unsigned int seed) {
  size_t gid =
      (__spirv_BuiltInGlobalInvocationId(0) * __spirv_BuiltInGlobalSize(1) *
       __spirv_BuiltInGlobalSize(2)) +
      (__spirv_BuiltInGlobalInvocationId(1) * __spirv_BuiltInGlobalSize(2)) +
      __spirv_BuiltInGlobalInvocationId(2);
  size_t global_size = __spirv_BuiltInGlobalSize(0) *
                       __spirv_BuiltInGlobalSize(1) *
                       __spirv_BuiltInGlobalSize(2);
  size_t gid1 =
      (global_size > RAND_NEXT_LEN) ? (gid & (RAND_NEXT_LEN - 1)) : gid;
  RAND_NEXT_ACC[gid1] = seed;
}

#if defined(_WIN32)
// Truncates a wide (16 or 32 bit) string (wstr) into an ASCII string (str).
// Any non-ASCII characters are replaced by question mark '?'.
static void __truncate_wchar_char_str(const wchar_t *wstr, char *str,
                                      size_t str_size) {
  str_size -= 1; // reserve for NULL terminator
  while (str_size > 0 && *wstr != L'\0') {
    wchar_t w = *wstr++;
    *str++ = (w > 0 && w < 127) ? (char)w : '?';
    str_size--;
  }
  *str = '\0';
}

DEVICE_EXTERN_C
void _wassert(const wchar_t *wexpr, const wchar_t *wfile, unsigned line) {
  // Paths and expressions that are longer than 256 characters are going to be
  // truncated.
  char file[256];
  __truncate_wchar_char_str(wfile, file, sizeof(file));
  char expr[256];
  __truncate_wchar_char_str(wexpr, expr, sizeof(expr));

  __devicelib_assert_fail(
      expr, file, line, /*func=*/nullptr, __spirv_BuiltInGlobalInvocationId(0),
      __spirv_BuiltInGlobalInvocationId(1),
      __spirv_BuiltInGlobalInvocationId(2), __spirv_BuiltInLocalInvocationId(0),
      __spirv_BuiltInLocalInvocationId(1), __spirv_BuiltInLocalInvocationId(2));
}
#else
DEVICE_EXTERN_C
void *malloc(size_t size) {
  return reinterpret_cast<void *>(0xEFEFEFEFEFEFEFEF);
}
DEVICE_EXTERN_C
void free(void *ptr) { return ; }
DEVICE_EXTERN_C
void __assert_fail(const char *expr, const char *file, unsigned int line,
                   const char *func) {
  __devicelib_assert_fail(
      expr, file, line, func, __spirv_BuiltInGlobalInvocationId(0),
      __spirv_BuiltInGlobalInvocationId(1),
      __spirv_BuiltInGlobalInvocationId(2), __spirv_BuiltInLocalInvocationId(0),
      __spirv_BuiltInLocalInvocationId(1), __spirv_BuiltInLocalInvocationId(2));
}

// In GCC-15, std::__glibcxx_assert_fail is added to do runtime check for some
// STL items such as std::array in debug mode, its behavior is same as assert,
// so just handle it in the same way as '__assert_fail'.
namespace std {
DEVICE_EXTERN_CPP
void __glibcxx_assert_fail(const char *file, int line, const char *func,
                           const char *cond) noexcept {
  __devicelib_assert_fail(
      cond, file, line, func, __spirv_BuiltInGlobalInvocationId(0),
      __spirv_BuiltInGlobalInvocationId(1),
      __spirv_BuiltInGlobalInvocationId(2), __spirv_BuiltInLocalInvocationId(0),
      __spirv_BuiltInLocalInvocationId(1), __spirv_BuiltInLocalInvocationId(2));
}
} // namespace std

#endif
#endif // __SPIR__ || __SPIRV__ || __NVPTX__ || __AMDGCN__
