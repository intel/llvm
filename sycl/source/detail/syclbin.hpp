//==--------------------- syclbin.hpp - SYCLBIN parser ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "detail/compiler.hpp"
#include "detail/device_binary_image.hpp"
#include "sycl/exception.hpp"
#include "llvm/Object/SYCLBIN.h"
#include "llvm/Support/PropertySetIO.h"

#include <algorithm>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace sycl {
inline namespace _V1 {

class device;

namespace detail {

// Helper class for managing both a SYCLBIN and binaries created from it,
// allowing existing infrastructure to better understand the contents of the
// SYCLBINs.
struct SYCLBINBinaries {
  // Delete copy-ctor to keep binaries unique and avoid costly copies of a
  // heavy structure.
  SYCLBINBinaries(const SYCLBINBinaries &) = delete;
  SYCLBINBinaries &operator=(const SYCLBINBinaries &) = delete;

  SYCLBINBinaries(SYCLBINBinaries &&) = default;
  SYCLBINBinaries &operator=(SYCLBINBinaries &&) = default;

  SYCLBINBinaries(const char *SYCLBINContent, size_t SYCLBINSize);

  ~SYCLBINBinaries() = default;

  std::vector<const RTDeviceBinaryImage *>
  getBestCompatibleImages(device_impl &Dev, bundle_state State);
  std::vector<const RTDeviceBinaryImage *>
  getBestCompatibleImages(devices_range Dev, bundle_state State);

  std::vector<const RTDeviceBinaryImage *>
  getNativeBinaryImages(device_impl &Dev);

  uint8_t getState() const {
    return static_cast<uint8_t>(
        ParsedSYCLBIN->GlobalMetadata
            ->at(llvm::util::PropertySet::key_type{"state"})
            .asUint32());
  }

private:
  std::vector<_sycl_device_binary_property_set_struct> &
  convertAbstractModuleProperties(llvm::object::SYCLBIN::AbstractModule &AM);

  std::unique_ptr<char[]> SYCLBINContentCopy = nullptr;
  std::unique_ptr<llvm::object::SYCLBIN> ParsedSYCLBIN;

  // Buffers for holding entries in the binary structs alive.
  std::vector<std::vector<_sycl_offload_entry_struct>> BinaryOffloadEntries;
  std::vector<std::vector<_sycl_device_binary_property_struct>>
      BinaryProperties;
  std::vector<std::vector<_sycl_device_binary_property_set_struct>>
      BinaryPropertySets;

  std::vector<sycl_device_binary_struct> DeviceBinaries;

  struct AbstractModuleDesc {
    size_t NumJITBinaries = 0;
    size_t NumNativeBinaries = 0;
    RTDeviceBinaryImage *JITBinaries;
    RTDeviceBinaryImage *NativeBinaries;
  };

  std::unique_ptr<AbstractModuleDesc[]> AbstractModuleDescriptors;
  std::unique_ptr<RTDeviceBinaryImage[]> BinaryImages;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
