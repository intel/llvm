// REQUIRES: gpu, level_zero
// https://github.com/intel/llvm/issues/12798
// UNSUPPORTED: windows
// RUN: env ONEAPI_DEVICE_SELECTOR="level_zero:*" sycl-ls 2>&1 | FileCheck --check-prefixes=CHECK-PI %s
// RUN: env SYCL_PREFER_UR=0 ONEAPI_DEVICE_SELECTOR="level_zero:*" sycl-ls 2>&1 | FileCheck --check-prefixes=CHECK-PI %s
// RUN: env SYCL_PI_TRACE=-1 SYCL_PREFER_UR=1 ONEAPI_DEVICE_SELECTOR="level_zero:*" sycl-ls 2>&1 | FileCheck --check-prefixes=CHECK-UR %s

// CHECK-PI: Intel(R) Level-Zero
// CHECK-UR: Intel(R) oneAPI Unified Runtime over Level-Zero

//==-- sycl-ls-unified-runtime.cpp ----- Test Unified Runtime platform  ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
