//==---------- defines.hpp ----- Preprocessor directives -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp> // for __SYCL_ID_QUERIES_FIT_...

#if __SYCL_ID_QUERIES_FIT_IN_INT__ && __has_builtin(__builtin_assume)
#include <climits>
#define __SYCL_ASSUME_INT(x) __builtin_assume((x) <= INT_MAX)
#else
#define __SYCL_ASSUME_INT(x)
#if __SYCL_ID_QUERIES_FIT_IN_INT__ && !__has_builtin(__builtin_assume)
#warning "No assumptions will be emitted due to no __builtin_assume available"
#endif
#endif

// FIXME Check for  __SYCL_DEVICE_ONLY__ can be removed if implementation of
// __has_attribute is fixed to consider LangOpts when generating attributes in
// tablegen.
#if __has_attribute(sycl_special_class) && (defined __SYCL_DEVICE_ONLY__)
#define __SYCL_SPECIAL_CLASS __attribute__((sycl_special_class))
#else
#define __SYCL_SPECIAL_CLASS
#endif

// FIXME Check for  __SYCL_DEVICE_ONLY__ can be removed if implementation of
// __has_attribute is fixed to consider LangOpts when generating attributes in
// tablegen.
#if __has_cpp_attribute(__sycl_detail__::sycl_type) &&                         \
    (defined __SYCL_DEVICE_ONLY__)
#define __SYCL_TYPE(x) [[__sycl_detail__::sycl_type(x)]]
#else
#define __SYCL_TYPE(x)
#endif

#if __has_cpp_attribute(clang::builtin_alias)
#define __SYCL_BUILTIN_ALIAS(x) [[clang::builtin_alias(x)]]
#else
#define __SYCL_BUILTIN_ALIAS(x)
#endif
