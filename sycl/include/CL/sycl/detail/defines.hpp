//==---------- defines.hpp ----- Preprocessor directives -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/defines_elementary.hpp>

#include <climits>

#if __SYCL_ID_QUERIES_FIT_IN_INT__ && __has_builtin(__builtin_assume)
#define __SYCL_ASSUME_INT(x) __builtin_assume((x) <= INT_MAX)
#else
#define __SYCL_ASSUME_INT(x)
#if __SYCL_ID_QUERIES_FIT_IN_INT__ && !__has_builtin(__builtin_assume)
#warning "No assumptions will be emitted due to no __builtin_assume available"
#endif
#endif
