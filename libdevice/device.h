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
#else   // __cplusplus
#define EXTERN_C
#endif  // __cplusplus

#ifdef CL_SYCL_LANGUAGE_VERSION
// FIXME: we have to include CL/sycl/detail/common.hpp here,
//        so that it undefined SYCL_EXTERNAL for host compilation.
//        I cannot do that now, because of the conflicting
//        definitions of _cl_motion_estimation_desc_intel in
//        cl_ext.h from Intel OpenCL SDK and
//        cl_ext_intel.h from Khronos. For some reason
//        OpenCL SDK prepends SYCL headers in the include order.
// #include "CL/sycl/detail/common.hpp"
#define DEVICE_EXTERNAL SYCL_EXTERNAL
#define IMPL_ENABLED __SYCL_DEVICE_ONLY__
#else   // CL_SYCL_LANGUAGE_VERSION
#define DEVICE_EXTERNAL
#define IMPL_ENABLED 0
#endif  // CL_SYCL_LANGUAGE_VERSION

#define DEVICE_EXTERN_C DEVICE_EXTERNAL EXTERN_C

#endif  // __LIBDEVICE_DEVICE_H__
