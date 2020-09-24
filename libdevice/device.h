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

#ifdef __SPIR__
#ifdef __SYCL_DEVICE_ONLY__
#define DEVICE_EXTERNAL SYCL_EXTERNAL __attribute__((weak))
#else // __SYCL_DEVICE_ONLY__
#define DEVICE_EXTERNAL __attribute__((weak))
#endif // __SYCL_DEVICE_ONLY__

#define DEVICE_EXTERN_C DEVICE_EXTERNAL EXTERN_C
#endif // __SPIR__

#endif // __LIBDEVICE_DEVICE_H__
