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
inline namespace _V1 {
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

// Inverse of createKernelArgMask: serialize a KernelArgMask back into the
// byte-array layout used by [SYCL/kernel param opt] property values:
//   - 8 little-endian bytes encoding size-in-bits;
//   - bits packed LSB-first into subsequent bytes.
// Used by the SYCLBIN serializer to round-trip runtime-tracked arg masks.
inline std::vector<std::uint8_t>
serializeKernelArgMask(const KernelArgMask &Mask) {
  const int NBytesForSize = 8;
  const int NBitsInElement = 8;
  const std::uint64_t SizeInBits = Mask.size();
  const std::uint64_t SizeInBytes =
      (SizeInBits + NBitsInElement - 1) / NBitsInElement;

  std::vector<std::uint8_t> Result(NBytesForSize + SizeInBytes, 0);
  for (int I = 0; I < NBytesForSize; ++I)
    Result[I] = static_cast<std::uint8_t>(SizeInBits >> (I * NBitsInElement));
  for (std::uint64_t I = 0; I < SizeInBits; ++I)
    if (Mask[I])
      Result[NBytesForSize + (I / NBitsInElement)] |=
          static_cast<std::uint8_t>(1u << (I % NBitsInElement));
  return Result;
}
} // namespace detail
} // namespace _V1
} // namespace sycl
