//==------- assert_happened.hpp - Assert signalling structure --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>

#include <cstdint>

#ifdef __SYCL_DEVICE_ONLY__
// Reads Flag of AssertHappened on device
SYCL_EXTERNAL __attribute__((weak)) extern "C" void
__devicelib_assert_read(void *);
#endif

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {
// NOTE Layout of this structure should be aligned with the one in
// libdevice/include/assert-happened.hpp
struct AssertHappened {
  int Flag = 0; // set to non-zero upon assert failure
  char Expr[256 + 1] = "";
  char File[256 + 1] = "";
  char Func[128 + 1] = "";

  int32_t Line = 0;

  uint64_t GID0 = 0;
  uint64_t GID1 = 0;
  uint64_t GID2 = 0;

  uint64_t LID0 = 0;
  uint64_t LID1 = 0;
  uint64_t LID2 = 0;
};
} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
