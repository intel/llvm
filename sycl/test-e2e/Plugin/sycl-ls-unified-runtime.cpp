// REQUIRES: TEMPORARY_DISABLED
//  Unified Runtime will soon be changing its reporting.
//
// RUN: env ONEAPI_DEVICE_SELECTOR="ext_oneapi_unified_runtime:*" sycl-ls --verbose 2>&1 %GPU_CHECK_PLACEHOLDER

// CHECK: 	Platforms: 1
// CHECK-NEXT:	Platform [#1]:
// CHECK-NEXT:      Version  : 1.3
// CHECK-NEXT:      Name     : Intel(R) oneAPI Unified Runtime over Level-Zero

//==-- sycl-ls-unified-runtime.cpp ----- Test Unified Runtime platform  ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
