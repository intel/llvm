//==--- device_assert_win.hpp - Redefinition of device assert on Windows---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#ifdef __SYCL_DEVICE_ONLY__
#include <CL/__spirv/spirv_vars.hpp>
#include <cassert>
#endif

#if defined(_WIN32) && defined(__SYCL_DEVICE_ONLY__) && defined(assert)
extern "C" __DPCPP_SYCL_EXTERNAL void
__devicelib_assert_fail(const char *, const char *, int32_t, const char *,
                        uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                        uint64_t);
#undef assert
#define assert(e)                                                              \
  (e) ? void(0)                                                                \
      : __devicelib_assert_fail(                                               \
            #e, __FILE__, __LINE__, nullptr, __spirv_GlobalInvocationId_x(),   \
            __spirv_GlobalInvocationId_y(), __spirv_GlobalInvocationId_z(),    \
            __spirv_LocalInvocationId_x(), __spirv_LocalInvocationId_y(),      \
            __spirv_LocalInvocationId_z());
#endif
