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

#if __has_attribute(always_inline)
#define __SYCL_ALWAYS_INLINE __attribute__((always_inline))
#else
#define __SYCL_ALWAYS_INLINE
#endif

#ifndef SYCL_EXTERNAL
#define SYCL_EXTERNAL
#endif

#ifndef __SYCL_ID_QUERIES_FIT_IN_INT__
#define __SYCL_ID_QUERIES_FIT_IN_INT__ 0
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
