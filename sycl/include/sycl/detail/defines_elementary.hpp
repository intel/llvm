//==--- defines_elementary.hpp ---- Preprocessor directives (simplified) ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#define __SYCL_INLINE_NAMESPACE(X) inline namespace X

#define __SYCL_INLINE_VER_NAMESPACE(X) inline namespace X

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
// The deprecated attribute is not supported in some situations(e.g. namespace)
// in C++14 mode
#if !defined(SYCL_DISABLE_DEPRECATION_WARNINGS) && __cplusplus >= 201703L
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
#if defined(__cplusplus) && __cplusplus >= 201703L &&                          \
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

// Stringify an argument to pass it in _Pragma directive below.
#ifndef __SYCL_STRINGIFY
#define __SYCL_STRINGIFY(x) #x
#endif // __SYCL_STRINGIFY

static_assert(__cplusplus >= 201703L,
              "DPCPP does not support C++ version earlier than C++17.");
