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
#define SYCL_EXT_ONEAPI_LOCAL_MEMORY 1
// As for SYCL_EXT_ONEAPI_MATRIX:
// 1- provides AOT initial implementation for AMX for the experimental matrix
// extension
// 2- provides JIT implementation (target agnostic) for the
// experimental matrix extension
#ifndef SYCL_EXT_ONEAPI_MATRIX
#define SYCL_EXT_ONEAPI_MATRIX 2
#endif
#define SYCL_EXT_INTEL_BF16_CONVERSION 1

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
