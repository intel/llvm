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

//#ifdef __SYCL_ENABLE_SYCL121_NAMESPACE
// Old SYCL1.2.1 namespace scheme
//#define __SYCL_NS_OPEN_1 __SYCL_INLINE_NAMESPACE(cl)
//#define __SYCL_NS_OPEN_2 namespace sycl
//#define __SYCL_NS cl::sycl
//#else
// New SYCL2020 friendly namespace scheme, defaulted to __v1
#define __SYCL_NS_OPEN_1 namespace __sycl_internal
#define __SYCL_NS_OPEN_2 namespace __v1
#define __SYCL_NS __sycl_internal::__v1
//#endif

#ifdef __SYCL_ENABLE_SYCL121_NAMESPACE

#define __SYCL_OPEN_NS()                                                       \
  __SYCL_NS_OPEN_1 {                                                           \
    __SYCL_NS_OPEN_2 {}                                                        \
  }                                                                            \
  namespace __sycl_ns = __SYCL_NS;                                             \
  __SYCL_INLINE_NAMESPACE(cl) {                                                \
  namespace sycl {                                                             \
  using namespace __SYCL_NS;                                                   \
  }                                                                            \
  }                                                                            \
  __SYCL_NS_OPEN_1 {                                                           \
    __SYCL_NS_OPEN_2

#define __SYCL_OPEN_NS_BUILTINS()                                              \
  __SYCL_NS_OPEN_1 {                                                           \
    __SYCL_NS_OPEN_2 {                                                         \
      namespace __host_std {}                                                  \
    }                                                                          \
  }                                                                            \
  __SYCL_INLINE_NAMESPACE(cl) {                                                \
  namespace __host_std = __SYCL_NS::__host_std;                                \
  }                                                                            \
  __SYCL_NS_OPEN_1 {                                                           \
    __SYCL_NS_OPEN_2

#define __SYCL_CLOSE_NS_BUILTINS() }

#else

// The macro:
// 1. Forward declares an empty "target" namespace for the alias
// 2. An alias which will be used to refer to "target" namepsace outside of
//    namespace itself
// 3. Opens "target" namespace
#define __SYCL_OPEN_NS()                                                       \
  __SYCL_NS_OPEN_1 {                                                           \
    __SYCL_NS_OPEN_2 {}                                                        \
  }                                                                            \
  namespace sycl {                                                             \
  using namespace __SYCL_NS;                                                   \
  }                                                                            \
  namespace __sycl_ns = __SYCL_NS;                                             \
  __SYCL_NS_OPEN_1 {                                                           \
    __SYCL_NS_OPEN_2

#define __SYCL_OPEN_NS_BUILTINS()                                              \
  __SYCL_NS_OPEN_1 {                                                           \
    __SYCL_NS_OPEN_2 {}                                                        \
  }                                                                            \
  namespace sycl {                                                             \
  using namespace __SYCL_NS;                                                   \
  }                                                                            \
  __SYCL_NS_OPEN_1 {                                                           \
    __SYCL_NS_OPEN_2

#define __SYCL_CLOSE_NS_BUILTINS() }

#endif

#define __SYCL_CLOSE_NS() }
