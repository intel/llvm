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

#ifndef __NVPTX__
#define RAND_NEXT_LEN 1024
DeviceGlobal<uint64_t[RAND_NEXT_LEN]> RandNext;
#endif

#if defined(__SPIR__) || defined(__SPIRV__) || defined(__NVPTX__)
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

#ifndef __NVPTX__

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
      (__spirv_BuiltInGlobalInvocationId.x * __spirv_BuiltInGlobalSize.y *
       __spirv_BuiltInGlobalSize.z) +
      (__spirv_BuiltInGlobalInvocationId.y * __spirv_BuiltInGlobalSize.z) +
      __spirv_BuiltInGlobalInvocationId.z;
  size_t global_size = __spirv_BuiltInGlobalSize.x *
                       __spirv_BuiltInGlobalSize.y *
                       __spirv_BuiltInGlobalSize.z;
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
      (__spirv_BuiltInGlobalInvocationId.x * __spirv_BuiltInGlobalSize.y *
       __spirv_BuiltInGlobalSize.z) +
      (__spirv_BuiltInGlobalInvocationId.y * __spirv_BuiltInGlobalSize.z) +
      __spirv_BuiltInGlobalInvocationId.z;
  size_t global_size = __spirv_BuiltInGlobalSize.x *
                       __spirv_BuiltInGlobalSize.y *
                       __spirv_BuiltInGlobalSize.z;
  size_t gid1 =
      (global_size > RAND_NEXT_LEN) ? (gid & (RAND_NEXT_LEN - 1)) : gid;
  RAND_NEXT_ACC[gid1] = seed;
}

#endif

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
      expr, file, line, /*func=*/nullptr, __spirv_GlobalInvocationId_x(),
      __spirv_GlobalInvocationId_y(), __spirv_GlobalInvocationId_z(),
      __spirv_LocalInvocationId_x(), __spirv_LocalInvocationId_y(),
      __spirv_LocalInvocationId_z());
}
#else
DEVICE_EXTERN_C
void __assert_fail(const char *expr, const char *file, unsigned int line,
                   const char *func) {
  __devicelib_assert_fail(
      expr, file, line, func, __spirv_GlobalInvocationId_x(),
      __spirv_GlobalInvocationId_y(), __spirv_GlobalInvocationId_z(),
      __spirv_LocalInvocationId_x(), __spirv_LocalInvocationId_y(),
      __spirv_LocalInvocationId_z());
}
#endif
#endif // __SPIR__ || __SPIRV__ || __NVPTX__
