//==------- assert_happened.hpp - Assert signalling structure --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/defines_elementary.hpp>

#ifdef __SYCL_DEVICE_ONLY__
// Reads Flag of AssertHappened on device
SYCL_EXTERNAL __attribute__((weak)) extern "C" int __devicelib_assert_read();
#endif

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
struct AssertHappened {
  int Flag = 0; // set to non-zero upon assert failure
};
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
