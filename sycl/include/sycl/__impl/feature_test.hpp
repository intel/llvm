//==---- feature_test.hpp - SYCL Feature Test Definitions -----*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
#pragma once

namespace __sycl_internal {
inline namespace __v1 {

// Feature test macro definitions

// TODO: Move these feature-test macros to compiler driver.
#define SYCL_EXT_INTEL_DEVICE_INFO 1
#define SYCL_EXT_ONEAPI_MATRIX 1

} // namespace sycl
} // namespace __sycl_internal
