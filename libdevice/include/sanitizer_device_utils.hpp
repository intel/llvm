//==-- sanitizer_device_utils.hpp - Declaration for sanitizer global var ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "spir_global_var.hpp"
#include <cstdint>

// Treat this header as system one to workaround frontend's restriction
#pragma clang system_header

#ifdef __SPIR__
#ifdef __SYCL_DEVICE_ONLY__

// declaration
extern SPIR_GLOBAL_VAR __SYCL_GLOBAL__ uintptr_t __AsanShadowMemoryGlobalStart;
extern SPIR_GLOBAL_VAR __SYCL_GLOBAL__ uintptr_t __AsanShadowMemoryGlobalEnd;
extern SPIR_GLOBAL_VAR __SYCL_GLOBAL__ uintptr_t __AsanShadowMemoryLocalStart;
extern SPIR_GLOBAL_VAR __SYCL_GLOBAL__ uintptr_t __AsanShadowMemoryLocalEnd;
extern SPIR_GLOBAL_VAR __SYCL_GLOBAL__ uintptr_t __AsanShadowMemoryPrivateStart;
extern SPIR_GLOBAL_VAR __SYCL_GLOBAL__ uintptr_t __AsanShadowMemoryPrivateEnd;

enum DeviceType : uintptr_t { UNKNOWN, CPU, GPU_PVC, GPU_DG2 };
extern SPIR_GLOBAL_VAR __SYCL_GLOBAL__ DeviceType __DeviceType;

#endif
#endif
