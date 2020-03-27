//==------------ devicelib.h - macros for devicelib wrappers ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __DEVICELIB_H__
#define __DEVICELIB_H__

#ifndef _WIN32
#include <features.h> // for GLIBC macro
#endif

// Devicelib wraps a system C or C++ library
#ifdef __GLIBC__
#define __DEVICELIB_GLIBC__ 1
#elif defined(_WIN32)
#define __DEVICELIB_MSLIBC__ 1
#endif

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

#ifdef __SYCL_DEVICE_ONLY__
#define __DEVICELIB_DEVICE_ONLY__ 1
#endif

#endif // __DEVICELIB_H__
