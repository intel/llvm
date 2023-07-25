//==--- defines_elementary.hpp ---- Preprocessor directives (simplified) ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

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

#ifdef SYCL_EXTERNAL
#define __DPCPP_SYCL_EXTERNAL SYCL_EXTERNAL
#else
#ifdef __SYCL_DEVICE_ONLY__
#define __DPCPP_SYCL_EXTERNAL __attribute__((sycl_device))
#else
#define __DPCPP_SYCL_EXTERNAL
#define SYCL_EXTERNAL
#endif
#endif

#ifndef __SYCL_ID_QUERIES_FIT_IN_INT__
#define __SYCL_ID_QUERIES_FIT_IN_INT__ 0
#endif

#ifndef __SYCL_DEPRECATED
#if !defined(SYCL2020_DISABLE_DEPRECATION_WARNINGS)
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

#ifndef __SYCL_WARN_IMAGE_ASPECT
#if !defined(SYCL_DISABLE_IMAGE_ASPECT_WARNING) && __has_attribute(diagnose_if)
#define __SYCL_WARN_IMAGE_ASPECT(aspect_param)                                   \
  __attribute__((diagnose_if(                                                    \
      aspect_param == aspect::image,                                             \
      "SYCL 2020 images are not supported on any devices. Consider using "       \
      "‘aspect::ext_intel_legacy_image’ instead. Disable this warning with " \
      "by defining SYCL_DISABLE_IMAGE_ASPECT_WARNING.",                          \
      "warning")))
#else
#define __SYCL_WARN_IMAGE_ASPECT(aspect)
#endif
#endif // __SYCL_WARN_IMAGE_ASPECT

#ifndef __SYCL_HAS_CPP_ATTRIBUTE
#if defined(__cplusplus) && defined(__has_cpp_attribute)
#define __SYCL_HAS_CPP_ATTRIBUTE(x) __has_cpp_attribute(x)
#else
#define __SYCL_HAS_CPP_ATTRIBUTE(x) 0
#endif
#endif

static_assert(__cplusplus >= 201703L,
              "DPCPP does not support C++ version earlier than C++17.");
