//==---------------- memcpy.hpp - SYCL memcpy --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {
inline void memcpy(void *Dst, const void *Src, size_t Size) {
  char *Destination = reinterpret_cast<char *>(Dst);
  const char *Source = reinterpret_cast<const char *>(Src);
  for (size_t I = 0; I < Size; ++I) {
    Destination[I] = Source[I];
  }
}
} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
