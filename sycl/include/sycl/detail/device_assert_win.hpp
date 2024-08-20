//==--- device_assert_win.hpp - Redefinition of device assert on Windows---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// Normal workflow for device assertions on Windows GPU does not work as
// expected, most likely because the SPV_EXT_relaxed_printf_string_address_space
// extension is not supported by IGC More specifically, the _wassert function
// seems to allocate the file and expression buffers on private memory and not
// constant memory which catches IGC by surprise. Therefore, we define this
// header file to explicitly redefine assert on Windows so that it directly
// calls __devicelib_assert_fail and not _wassert.
// TODO: Delete this header file and its inclusion in <sycl/sycl.hpp> once the
// extension is supported by IGC.

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
