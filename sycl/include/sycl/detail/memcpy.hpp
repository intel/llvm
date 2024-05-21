//==---------------- memcpy.hpp - SYCL memcpy --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstring>

namespace sycl {
inline namespace _V1 {
namespace detail {
inline void memcpy(void *Dst, const void *Src, size_t Size) {
#ifdef __SYCL_DEVICE_ONLY__
  __builtin_memcpy(Dst, Src, Size);
#else
  std::memcpy(Dst, Src, Size);
#endif
}
} // namespace detail
} // namespace _V1
} // namespace sycl
