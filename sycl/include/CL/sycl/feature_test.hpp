//==---- feature_test.hpp - SYCL Feature Test Definitions -----*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
#pragma once

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// Feature test macro definitions

// TODO: Move these feature-test macros to compiler driver.
#define SYCL_EXT_INTEL_DEVICE_INFO 2
#define SYCL_EXT_ONEAPI_MATRIX 1
#define SYCL_EXT_ONEAPI_MAX_GLOBAL_NUMBER_WORK_GROUPS 1
#define SYCL_EXT_ONEAPI_MAX_NUMBER_WORK_GROUPS 1

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
