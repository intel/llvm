//==--- msvc_wrapper.cpp - wrappers for Microsoft C library functions ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef __SYCL_DEVICE_ONLY__
#include "wrapper.h"

#include <CL/__spirv/spirv_vars.hpp> // for __spirv_BuiltInGlobalInvocationId,
                                     //     __spirv_BuiltInLocalInvocationId

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

extern "C" SYCL_EXTERNAL
void _wassert(const wchar_t *wexpr, const wchar_t *wfile, unsigned line) {
  // Paths and expressions that are longer than 256 characters are going to be
  // truncated.
  char file[256];
  __truncate_wchar_char_str(wfile, file, sizeof(file));
  char expr[256];
  __truncate_wchar_char_str(wexpr, expr, sizeof(expr));

  __devicelib_assert_fail(expr, file, line, /*func=*/nullptr,
                          __spirv_BuiltInGlobalInvocationId.x,
                          __spirv_BuiltInGlobalInvocationId.y,
                          __spirv_BuiltInGlobalInvocationId.z,
                          __spirv_BuiltInLocalInvocationId.x,
                          __spirv_BuiltInLocalInvocationId.y,
                          __spirv_BuiltInLocalInvocationId.z);
}
#endif // __SYCL_DEVICE_ONLY__
