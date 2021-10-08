//==------ crt_wrapper.cpp - wrappers for libc internal functions ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "wrapper.h"

#ifdef __SPIR__
DEVICE_EXTERN_C
void *memcpy(void *dest, const void *src, size_t n) {
  return __devicelib_memcpy(dest, src, n);
}

DEVICE_EXTERN_C
void *memset(void *dest, int c, size_t n) {
  return __devicelib_memset(dest, c, n);
}

DEVICE_EXTERN_C
int memcmp(const void *s1, const void *s2, size_t n) {
  return __devicelib_memcmp(s1, s2, n);
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
#endif // __SPIR__
