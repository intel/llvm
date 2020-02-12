//==---------- defines.hpp ----- Preprocessor directives -------------------==//
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

#if __has_attribute(always_inline)
#define ALWAYS_INLINE __attribute__((always_inline))
#else
#define ALWAYS_INLINE
#endif

#ifndef SYCL_EXTERNAL
#define SYCL_EXTERNAL
#endif

#if __cplusplus >= 201402
  #define __SYCL_DEPRECATED__                                                  \
    [[deprecated("Replaced by in_order queue property")]]
#elif !defined _MSC_VER
  #define __SYCL_DEPRECATED__ __attribute__                                    \
    ((deprecated("Replaced by in_order queue property")))
#else
  #define __SYCL_DEPRECATED__
#endif
