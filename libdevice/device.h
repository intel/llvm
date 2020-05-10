//==--- device.h - device definitions ------------------------*- C++ -*-----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LIBDEVICE_DEVICE_H__
#define __LIBDEVICE_DEVICE_H__

#ifdef __cplusplus
#define EXTERN_C extern "C"
#else // __cplusplus
#define EXTERN_C
#endif // __cplusplus

#ifdef CL_SYCL_LANGUAGE_VERSION
#ifndef SYCL_EXTERNAL
#define SYCL_EXTERNAL
#endif // SYCL_EXTERNAL

#ifdef __SYCL_DEVICE_ONLY__
#define DEVICE_EXTERNAL SYCL_EXTERNAL __attribute__((weak))
#else // __SYCL_DEVICE_ONLY__
#define DEVICE_EXTERNAL static
#undef EXTERN_C
#define EXTERN_C
#endif // __SYCL_DEVICE_ONLY__
#else  // CL_SYCL_LANGUAGE_VERSION
#define DEVICE_EXTERNAL
#endif // CL_SYCL_LANGUAGE_VERSION

#define DEVICE_EXTERN_C DEVICE_EXTERNAL EXTERN_C

// We need the following header to ensure the definition of all spirv variables
// required by the wrapper libraries.
#include "spirv_vars.hpp"

#endif // __LIBDEVICE_DEVICE_H__
