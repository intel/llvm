//==--- glibc_wrapper.cpp - wrappers for Glibc internal functions ----------==//
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

extern "C" SYCL_EXTERNAL
void __assert_fail(const char *expr, const char *file,
                   unsigned int line, const char *func) {
  __devicelib_assert_fail(expr, file, line, func,
                          __spirv_GlobalInvocationId_x(),
                          __spirv_GlobalInvocationId_y(),
                          __spirv_GlobalInvocationId_z(),
                          __spirv_LocalInvocationId_x(),
                          __spirv_LocalInvocationId_y(),
                          __spirv_LocalInvocationId_z());
}
#endif // __SYCL_DEVICE_ONLY__
