//==--- wrapper.h - declarations for devicelib functions -----*- C++ -*-----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LIBDEVICE_WRAPPER_H__
#define __LIBDEVICE_WRAPPER_H__

#include "device.h"
#include <cstddef>
#include <cstdint>

DEVICE_EXTERNAL size_t __spirv_GlobalInvocationId_x();
DEVICE_EXTERNAL size_t __spirv_GlobalInvocationId_y();
DEVICE_EXTERNAL size_t __spirv_GlobalInvocationId_z();

DEVICE_EXTERNAL size_t __spirv_LocalInvocationId_x();
DEVICE_EXTERNAL size_t __spirv_LocalInvocationId_y();
DEVICE_EXTERNAL size_t __spirv_LocalInvocationId_z();

DEVICE_EXTERNAL int
__spirv_ocl_printf(const __attribute__((opencl_constant)) char *fmt, ...);

DEVICE_EXTERN_C
void __devicelib_assert_fail(const char *expr, const char *file, int32_t line,
                             const char *func, uint64_t gid0, uint64_t gid1,
                             uint64_t gid2, uint64_t lid0, uint64_t lid1,
                             uint64_t lid2);

#endif // __LIBDEVICE_WRAPPER_H__
