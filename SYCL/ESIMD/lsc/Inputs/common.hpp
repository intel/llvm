//==------- common.hpp - DPC++ ESIMD on-device test ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdlib.h>

template <int case_num> class KernelID;

template <typename T> T get_rand() {
  T v = rand();
  if constexpr (sizeof(T) > 4)
    v = (v << 32) | rand();
  return v;
}
