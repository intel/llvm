//==----------- kernel_arg_mask.hpp - SYCL KernelArgMask -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <detail/device_binary_image.hpp>
#include <vector>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {
using KernelArgMask = std::vector<bool>;
inline KernelArgMask createKernelArgMask(const ByteArray &Bytes) {
  const int NBytesForSize = 8;
  const int NBitsInElement = 8;
  std::uint64_t SizeInBits = 0;

  KernelArgMask Result;
  for (int I = 0; I < NBytesForSize; ++I)
    SizeInBits |= static_cast<std::uint64_t>(Bytes[I]) << I * NBitsInElement;

  Result.reserve(SizeInBits);
  for (std::uint64_t I = 0; I < SizeInBits; ++I) {
    std::uint8_t Byte = Bytes[NBytesForSize + (I / NBitsInElement)];
    Result.push_back(Byte & (1 << (I % NBitsInElement)));
  }
  return Result;
}
} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
