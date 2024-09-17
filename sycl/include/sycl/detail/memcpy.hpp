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
// Using "memcpy_no_adl" function name instead of "memcpy" to prevent
// ambiguity with libc's memcpy. Even though they are in a different
// namespace, due to ADL, compiler may lookup "memcpy" symbol in the
// sycl::detail namespace, like in the following code:
//    sycl::vec<int , 1> a, b;
//    memcpy(&a, &b, sizeof(sycl::vec<int , 1>));
inline void memcpy_no_adl(void *Dst, const void *Src, size_t Size) {
#ifdef __SYCL_DEVICE_ONLY__
  __builtin_memcpy(Dst, Src, Size);
#else
  std::memcpy(Dst, Src, Size);
#endif
}
} // namespace detail
} // namespace _V1
} // namespace sycl
