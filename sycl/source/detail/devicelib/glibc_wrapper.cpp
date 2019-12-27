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
                          __spirv_BuiltInGlobalInvocationId.x,
                          __spirv_BuiltInGlobalInvocationId.y,
                          __spirv_BuiltInGlobalInvocationId.z,
                          __spirv_BuiltInLocalInvocationId.x,
                          __spirv_BuiltInLocalInvocationId.y,
                          __spirv_BuiltInLocalInvocationId.z);
}
#endif // __SYCL_DEVICE_ONLY__
