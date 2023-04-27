// REQUIRES: gpu, level_zero
// RUN: env ONEAPI_DEVICE_SELECTOR="ext_oneapi_level_zero:*" sycl-ls --verbose 2>&1 | FileCheck --check-prefixes=CHECK-PI %s
// RUN: env SYCL_PREFER_UR=0 ONEAPI_DEVICE_SELECTOR="ext_oneapi_level_zero:*" sycl-ls --verbose 2>&1 | FileCheck --check-prefixes=CHECK-PI %s
// RUN: env SYCL_PREFER_UR=1 ONEAPI_DEVICE_SELECTOR="ext_oneapi_level_zero:*" sycl-ls --verbose 2>&1 | FileCheck --check-prefixes=CHECK-UR %s

// CHECK-PI: 		Platforms: 1
// CHECK-PI-NEXT:	Platform [#1]:
// CHECK-PI-NEXT:      	Version  : 1.3
// CHECK-PI-NEXT:      	Name     : Intel(R) Level-Zero

// CHECK-UR: 		Platforms: 1
// CHECK-UR-NEXT:	Platform [#1]:
// CHECK-UR-NEXT:      	Version  : 1.3
// CHECK-UR-NEXT:      	Name     : Intel(R) oneAPI Unified Runtime over Level-Zero

//==-- sycl-ls-unified-runtime.cpp ----- Test Unified Runtime platform  ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
