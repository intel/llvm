//==--------- devicelib-assert.h - wrapper definition for C assert ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "devicelib.h"
#include <cstddef>
#include <cstdint>

DEVICE_EXTERNAL size_t __spirv_GlobalInvocationId_x();
DEVICE_EXTERNAL size_t __spirv_GlobalInvocationId_y();
DEVICE_EXTERNAL size_t __spirv_GlobalInvocationId_z();

DEVICE_EXTERNAL size_t __spirv_LocalInvocationId_x();
DEVICE_EXTERNAL size_t __spirv_LocalInvocationId_y();
DEVICE_EXTERNAL size_t __spirv_LocalInvocationId_z();

DEVICE_EXTERN_C
void __devicelib_assert_fail(const char *expr, const char *file, int32_t line,
                             const char *func, uint64_t gid0, uint64_t gid1,
                             uint64_t gid2, uint64_t lid0, uint64_t lid1,
                             uint64_t lid2);


#ifdef __DEVICELIB_GLIBC__
EXTERN_C inline void __assert_fail(const char *expr, const char *file,
                                   unsigned int line, const char *func) {
  __devicelib_assert_fail(
      expr, file, line, func, __spirv_GlobalInvocationId_x(),
      __spirv_GlobalInvocationId_y(), __spirv_GlobalInvocationId_z(),
      __spirv_LocalInvocationId_x(), __spirv_LocalInvocationId_y(),
      __spirv_LocalInvocationId_z());

}
#elif defined(__DEVICELIB_MSLIBC__)
// Truncates a wide (16 or 32 bit) string (wstr) into an ASCII string (str).
// Any non-ASCII characters are replaced by question mark '?'.
static inline void __truncate_wchar_char_str(const wchar_t *wstr, char *str,
                                      size_t str_size) {
  str_size -= 1; // reserve for NULL terminator
  while (str_size > 0 && *wstr != L'\0') {
    wchar_t w = *wstr++;
    *str++ = (w > 0 && w < 127) ? (char)w : '?';
    str_size--;
  }
  *str = '\0';
}

EXTERN_C inline void _wassert(const wchar_t *wexpr, const wchar_t *wfile,
                              unsigned line) {
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

#endif
