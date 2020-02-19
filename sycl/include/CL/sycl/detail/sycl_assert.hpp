//==----------- sycl_assert.hpp - SYCL standard header file ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef __SYCL_ENABLE_ASSERTIONS__
#define __SYCL_ASSERT(X) assert(X)
#else  // #ifdef __SYCL_ENABLE_ASSERTIONS__
#define __SYCL_ASSERT(X, ...)
#endif // #ifdef __SYCL_ENABLE_ASSERTIONS__

