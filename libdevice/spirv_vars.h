//==---------- spirv_vars.h --- SPIRV variables --------------*- C++ -*-----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LIBDEVICE_SPIRV_VARS_H
#define __LIBDEVICE_SPIRV_VARS_H

#include "device.h"
#include <cstddef>
#include <cstdint> // for uint32_t

#ifdef __SYCL_DEVICE_ONLY__

// SPIR-V built-in variables mapped to function call.

DEVICE_EXTERNAL size_t __spirv_BuiltInGlobalInvocationId(int);
DEVICE_EXTERNAL size_t __spirv_BuiltInGlobalSize(int);
DEVICE_EXTERNAL size_t __spirv_BuiltInGlobalOffset(int);
DEVICE_EXTERNAL size_t __spirv_BuiltInNumWorkgroups(int);
DEVICE_EXTERNAL size_t __spirv_BuiltInWorkgroupSize(int);
DEVICE_EXTERNAL size_t __spirv_BuiltInWorkgroupId(int);
DEVICE_EXTERNAL size_t __spirv_BuiltInLocalInvocationId(int);
DEVICE_EXTERNAL size_t __spirv_BuiltInGlobalLinearId();

DEVICE_EXTERNAL uint32_t __spirv_BuiltInSubgroupSize();
DEVICE_EXTERNAL uint32_t __spirv_BuiltInSubgroupMaxSize();
DEVICE_EXTERNAL uint32_t __spirv_BuiltInNumSubgroups();
DEVICE_EXTERNAL uint32_t __spirv_BuiltInSubgroupId();
DEVICE_EXTERNAL uint32_t __spirv_BuiltInSubgroupLocalInvocationId();

#endif // __SYCL_DEVICE_ONLY__

#endif // __LIBDEVICE_SPIRV_VARS_H
