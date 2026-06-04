//==--------------------- syclbin.hpp - SYCLBIN parser ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "detail/device_binary_image.hpp"

#include <memory>
#include <vector>

namespace sycl {
inline namespace _V1 {
namespace detail {

// Helper class wrapping a parsed SYCLBIN and the binaries derived from it.
//
// The on-disk SYCLBIN format and its in-memory parser live in LLVMObject
// (llvm::object::SYCLBIN). To keep LLVM types out of the SYCL runtime's
// public/internal headers, the concrete parser state is held by an opaque
// Impl and only sycl-native types appear in this interface.
struct SYCLBINBinaries {
  SYCLBINBinaries(const SYCLBINBinaries &) = delete;
  SYCLBINBinaries &operator=(const SYCLBINBinaries &) = delete;
  SYCLBINBinaries(SYCLBINBinaries &&) noexcept;
  SYCLBINBinaries &operator=(SYCLBINBinaries &&) noexcept;

  SYCLBINBinaries(const char *SYCLBINContent, size_t SYCLBINSize);
  ~SYCLBINBinaries();

  std::vector<const RTDeviceBinaryImage *>
  getBestCompatibleImages(device_impl &Dev, bundle_state State);
  std::vector<const RTDeviceBinaryImage *>
  getBestCompatibleImages(devices_range Dev, bundle_state State);

  std::vector<const RTDeviceBinaryImage *>
  getNativeBinaryImages(device_impl &Dev);

  uint8_t getState() const;

private:
  struct Impl;
  std::unique_ptr<Impl> PImpl;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
