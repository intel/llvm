//==----------- builtins.hpp - SYCL built-in functions ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>

#include <sycl/detail/builtins/builtins.hpp>

#ifdef __SYCL_DEVICE_ONLY__
extern "C" {

extern __DPCPP_SYCL_EXTERNAL_LIBC void *memcpy(void *dest, const void *src,
                                               size_t n);
extern __DPCPP_SYCL_EXTERNAL_LIBC void *memset(void *dest, int c, size_t n);
extern __DPCPP_SYCL_EXTERNAL_LIBC int memcmp(const void *s1, const void *s2,
                                             size_t n);
extern __DPCPP_SYCL_EXTERNAL_LIBC size_t strlen(const char *s);
extern __DPCPP_SYCL_EXTERNAL_LIBC char *strcpy(char *dest, const char *src);
extern __DPCPP_SYCL_EXTERNAL_LIBC char *strncpy(char *dest, const char *src,
                                                size_t n);
extern __DPCPP_SYCL_EXTERNAL_LIBC int strcmp(const char *s1, const char *s2);
extern __DPCPP_SYCL_EXTERNAL_LIBC int strncmp(const char *s1, const char *s2,
                                              size_t n);
extern __DPCPP_SYCL_EXTERNAL_LIBC int rand();
extern __DPCPP_SYCL_EXTERNAL_LIBC void srand(unsigned int seed);
}
#ifdef __GLIBC__
namespace std {
extern __DPCPP_SYCL_EXTERNAL_LIBC void
__glibcxx_assert_fail(const char *file, int line, const char *func,
                      const char *cond) noexcept;
} // namespace std
extern "C" {
extern __DPCPP_SYCL_EXTERNAL_LIBC void __assert_fail(const char *expr,
                                                     const char *file,
                                                     unsigned int line,
                                                     const char *func);
}
#elif defined(_WIN32)
extern "C" {
// TODO: documented C runtime library APIs must be recognized as
//       builtins by FE. This includes _dpcomp, _dsign, _dtest,
//       _fdpcomp, _fdsign, _fdtest, _hypotf, _wassert.
//       APIs used by STL, such as _Cosh, are undocumented, even though
//       they are open-sourced. Recognizing them as builtins is not
//       straightforward currently.
extern __DPCPP_SYCL_EXTERNAL_LIBC void
_wassert(const wchar_t *wexpr, const wchar_t *wfile, unsigned line);
}
#endif
#endif // __SYCL_DEVICE_ONLY__
