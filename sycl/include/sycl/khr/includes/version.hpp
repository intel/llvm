//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __SYCL_KHR_INCLUDES_VERSION
#define __SYCL_KHR_INCLUDES_VERSION

// SYCL_LANGUAGE_VERSION is currently defined by the compiler.
// We emit a warning if <sycl/sycl.hpp> is used without -fsycl flag, i.e. no
// guarantees are provided about the completeness of the headers in this mode.
// The same applies to the khr_includes implementation. If those headers are
// used outside of the -fsycl mode, then no correctness guarantee is provided.
// TODO: Implement a similar warning about using those headers outside of -fsycl
//       mode.

/// We support everything from the specification
#define SYCL_FEATURE_SET_FULL 1

/// This macro must be defined to 1 when SYCL implementation allows user
/// applications to explicitly declare certain class types as device copyable
/// by adding specializations of is_device_copyable type trait class.
#define SYCL_DEVICE_COPYABLE 1

// SYCL_EXTERNAL definition
#include <sycl/detail/defines_elementary.hpp>

// Extension-provided macro
#include <sycl/feature_test.hpp>

#endif // __SYCL_KHR_INCLUDES_VERSION
