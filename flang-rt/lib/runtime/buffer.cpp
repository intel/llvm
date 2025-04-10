//===-- lib/runtime/buffer.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang-rt/runtime/buffer.h"
#include <algorithm>

namespace Fortran::runtime::io {
RT_OFFLOAD_API_GROUP_BEGIN

// Here's a very old trick for shifting circular buffer data cheaply
// without a need for a temporary array.
void LeftShiftBufferCircularly(
    char *buffer, std::size_t bytes, std::size_t shift) {
  // Assume that we start with "efgabcd" and the left shift is 3.
  RT_DIAG_PUSH
  RT_DIAG_DISABLE_CALL_HOST_FROM_DEVICE_WARN
  std::reverse(buffer, buffer + shift); // "gfeabcd"
  std::reverse(buffer, buffer + bytes); // "dcbaefg"
  std::reverse(buffer, buffer + bytes - shift); // "abcdefg"
  RT_DIAG_POP
}

RT_OFFLOAD_API_GROUP_END
} // namespace Fortran::runtime::io
