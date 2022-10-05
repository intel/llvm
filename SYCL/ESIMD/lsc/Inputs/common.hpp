//==------- common.hpp - DPC++ ESIMD on-device test ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdlib.h>
#include <sycl/bit_cast.hpp>

template <int case_num> class KernelID;

template <typename T> T get_rand() {
  using Tuint = std::conditional_t<
      sizeof(T) == 1, uint8_t,
      std::conditional_t<
          sizeof(T) == 2, uint16_t,
          std::conditional_t<sizeof(T) == 4, uint32_t,
                             std::conditional_t<sizeof(T) == 8, uint64_t, T>>>>;
  Tuint v = rand();
  if constexpr (sizeof(Tuint) > 4)
    v = (v << 32) | rand();
  return sycl::bit_cast<T>(v);
}
