//==------------ hip_definitions.hpp - SYCL ROCM backend ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// HIP backend specific options
// TODO: Use values that won't overlap with others

// Mem Object info: Retrieve the raw HIP pointer from a cl_mem
#define __SYCL_PI_HIP_RAW_POINTER (0xFF01)
// Context creation: Use a primary HIP context instead of a custom one by
//                   providing a property value of PI_TRUE for the following
//                   property ID.
#define __SYCL_PI_CONTEXT_PROPERTIES_HIP_PRIMARY (0xFF02)

// PI Command Queue using Default stream
#define __SYCL_PI_HIP_USE_DEFAULT_STREAM (0xFF03)
// PI Command queue will sync with default stream
#define __SYCL_PI_HIP_SYNC_WITH_DEFAULT (0xFF04)
