//==------------ cuda_definitions.hpp - SYCL CUDA backend ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// CUDA backend specific options
// TODO: Use values that won't overlap with others

// Mem Object info: Retrieve the raw CUDA pointer from a cl_mem
#define __SYCL_PI_CUDA_RAW_POINTER (0xFF01)
// Context creation: Use a primary CUDA context instead of a custom one by
//                   providing a property value of PI_TRUE for the following
//                   property ID.
#define __SYCL_PI_CONTEXT_PROPERTIES_CUDA_PRIMARY (0xFF02)

// PI Command Queue using Default stream
#define __SYCL_PI_CUDA_USE_DEFAULT_STREAM (0xFF03)
// PI Command queue will sync with default stream
#define __SYCL_PI_CUDA_SYNC_WITH_DEFAULT (0xFF04)
