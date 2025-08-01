//== ----------------<assert.h> wrapper around STL--------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Must not be guarded. C++ standard says the macro assert is redefined
// according to the current state of NDEBUG each time that <assert.h> is
// included.

#if defined(__has_include_next)
#include_next <assert.h>
#else
#include <../ucrt/assert.h>
#endif

extern "C++" {
#ifdef __SYCL_DEVICE_ONLY__
#include <sycl/__spirv/spirv_vars.hpp>

// Device assertions on Windows do not work properly so we define these wrappers
// around the STL assertion headers cassert and assert.h where we redefine
// the assert macro to call __devicelib_assert_fail directly and bypass
// _wassert.
#if defined(_WIN32) && defined(assert)
extern "C" __DPCPP_SYCL_EXTERNAL void
__devicelib_assert_fail(const char *, const char *, int32_t, const char *,
                        uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                        uint64_t);
#undef assert
#if defined(NDEBUG)
#define assert(e) ((void)0)
#else
#define assert(e)                                                              \
  ((e) ? void(0)                                                               \
       : __devicelib_assert_fail(#e, __FILE__, __LINE__, nullptr,              \
                                 __spirv_BuiltInGlobalInvocationId(0),         \
                                 __spirv_BuiltInGlobalInvocationId(1),         \
                                 __spirv_BuiltInGlobalInvocationId(2),         \
                                 __spirv_BuiltInLocalInvocationId(0),          \
                                 __spirv_BuiltInLocalInvocationId(1),          \
                                 __spirv_BuiltInLocalInvocationId(2)))
#endif
#endif
#endif
}
