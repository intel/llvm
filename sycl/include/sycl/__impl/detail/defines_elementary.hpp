//==--- defines_elementary.hpp ---- Preprocessor directives (simplified) ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

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
#endif // __has_builtin

#ifndef __SYCL_ALWAYS_INLINE
#if __has_attribute(always_inline)
#define __SYCL_ALWAYS_INLINE __attribute__((always_inline))
#else
#define __SYCL_ALWAYS_INLINE
#endif
#endif // __SYCL_ALWAYS_INLINE

#ifndef SYCL_EXTERNAL
#define SYCL_EXTERNAL
#endif

#ifndef __SYCL_ID_QUERIES_FIT_IN_INT__
#define __SYCL_ID_QUERIES_FIT_IN_INT__ 0
#endif

#ifndef __SYCL_DEPRECATED
#ifndef SYCL_DISABLE_DEPRECATION_WARNINGS
#define __SYCL_DEPRECATED(message) [[deprecated(message)]]
#else // SYCL_DISABLE_DEPRECATION_WARNINGS
#define __SYCL_DEPRECATED(message)
#endif // SYCL_DISABLE_DEPRECATION_WARNINGS
#endif // __SYCL_DEPRECATED

#ifndef __SYCL2020_DEPRECATED
#if SYCL_LANGUAGE_VERSION >= 202001 &&                                         \
    !defined(SYCL2020_DISABLE_DEPRECATION_WARNINGS)
#define __SYCL2020_DEPRECATED(message) __SYCL_DEPRECATED(message)
#else
#define __SYCL2020_DEPRECATED(message)
#endif
#endif // __SYCL2020_DEPRECATED

#ifndef __SYCL_INLINE_CONSTEXPR
// inline constexpr is a C++17 feature
#if __cplusplus >= 201703L
#define __SYCL_INLINE_CONSTEXPR inline constexpr
#else
#define __SYCL_INLINE_CONSTEXPR static constexpr
#endif
#endif // __SYCL_INLINE_CONSTEXPR

#ifndef __SYCL_HAS_CPP_ATTRIBUTE
#if defined(__cplusplus) && defined(__has_cpp_attribute)
#define __SYCL_HAS_CPP_ATTRIBUTE(x) __has_cpp_attribute(x)
#else
#define __SYCL_HAS_CPP_ATTRIBUTE(x) 0
#endif
#endif

#ifndef __SYCL_FALLTHROUGH
#if defined(__cplusplus) && __cplusplus > 201402L &&                           \
    __SYCL_HAS_CPP_ATTRIBUTE(fallthrough)
#define __SYCL_FALLTHROUGH [[fallthrough]]
#elif __SYCL_HAS_CPP_ATTRIBUTE(gnu::fallthrough)
#define __SYCL_FALLTHROUGH [[gnu::fallthrough]]
#elif __has_attribute(fallthrough)
#define __SYCL_FALLTHROUGH __attribute__((fallthrough))
#elif __SYCL_HAS_CPP_ATTRIBUTE(clang::fallthrough)
#define __SYCL_FALLTHROUGH [[clang::fallthrough]]
#else
#define __SYCL_FALLTHROUGH
#endif
#endif // __SYCL_FALLTHROUGH

// define __SYCL_WARNING convenience macro to report compiler warnings
#if defined(__GNUC__)
#define __SYCL_GCC_PRAGMA(x) _Pragma(#x)
#define __SYCL_WARNING(msg) __SYCL_GCC_PRAGMA(GCC warning msg)
#elif defined(_MSC_VER) && !defined(__clang__)
#define __SYCL_QUOTE1(x) #x
#define __SYCL_QUOTE(x) __SYCL_QUOTE1(x)
#define __SYCL_SRC_LOC __FILE__ ":" __SYCL_QUOTE(__LINE__)
#define __SYCL_WARNING(msg) __pragma(message(__SYCL_SRC_LOC " warning: " msg))
#else // clang et. al.
// clang emits "warning:" in the message pragma output
#define __SYCL_WARNING(msg) __pragma(message(msg))
#endif // __GNUC__
