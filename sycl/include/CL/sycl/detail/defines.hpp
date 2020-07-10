//==---------- defines.hpp ----- Preprocessor directives -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <climits>

#ifndef __SYCL_DISABLE_NAMESPACE_INLINE__
#define __SYCL_INLINE_NAMESPACE(X) inline namespace X
#else
#define __SYCL_INLINE_NAMESPACE(X) namespace X
#endif // __SYCL_DISABLE_NAMESPACE_INLINE__

#ifndef __has_attribute
#define __has_attribute(x) 0
#endif

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#if __has_attribute(always_inline)
#define ALWAYS_INLINE __attribute__((always_inline))
#else
#define ALWAYS_INLINE
#endif

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif // __has_builtin

#ifndef SYCL_EXTERNAL
#define SYCL_EXTERNAL
#endif

#if defined(__SYCL_ID_QUERIES_FIT_IN_INT__) && __has_builtin(__builtin_assume)
#define __SYCL_ASSUME_INT(x) __builtin_assume((x) <= INT_MAX)
#else
#define __SYCL_ASSUME_INT(x)
#if defined(__SYCL_ID_QUERIES_FIT_IN_INT__) && !__has_builtin(__builtin_assume)
#warning "No assumptions will be emitted due to no __builtin_assume available"
#endif
#endif

#ifdef _WIN32
#define __SYCL_DEPRECATED(message) __declspec(deprecated(message))
#else
#define __SYCL_DEPRECATED(message) __attribute__((deprecated(message)))
#endif

// inline constexpr is a C++17 feature
#if __cplusplus >= 201703L
#define __SYCL_INLINE_CONSTEXPR inline constexpr
#else
#define __SYCL_INLINE_CONSTEXPR static constexpr
#endif

#if __has_attribute(sycl_special_class)
#define __SYCL_SPECIAL_CLASS(kind) __attribute__((sycl_special_class(kind)))
#else
#define __SYCL_SPECIAL_CLASS(kind)
#endif
