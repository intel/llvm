//==-------- version --- SYCL preprocessor definitions ---------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __SYCL_KHR_INCLUDES_VERSION
#define __SYCL_KHR_INCLUDES_VERSION

// SYCL_LANGUAGE_VERSION is currently defined by the compiler. If we ever change
// that, then it must be defined in this header (directly, or indirectly)
#ifndef SYCL_LANGUAGE_VERSION
// Can't build sycl-ls - it seems like it uses sycl.hpp with a 3rd-party
// compiler.
// Ideally, this should be set by the compiler, because it allows to specify
// a version.
// However, we may need to have some fallback here if someone includes SYCL
// headers into their host applications and use 3rd-party compilers.
// #error "SYCL_LANGUAGE_VERSION is not defined, please report this as a bug"
#endif

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
