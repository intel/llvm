//==----------------- device_global_map_entry.hpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <unordered_map>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

// Forward declaration
class device_impl;

struct DeviceGlobalMapEntry {
  // Pointer to the device_global on host.
  void *MDeviceGlobalPtr;
  // Size of the underlying type in the device_global.
  std::uint32_t MDeviceGlobalTSize;
  // True if the device_global has been decorated with device_image_scope
  bool MIsDeviceImageScopeDecorated;
  // Map between devices and corresponding USM allocations for the
  // device_global. This should always be empty if MIsDeviceImageScopeDecorated
  // is true.
  std::unordered_map<std::shared_ptr<device_impl>, void *> MDeviceToUSMPtrMap;

  // Constructor only initializes with the pointer to the device_global as the
  // additional information is loaded after.
  DeviceGlobalMapEntry(void *DeviceGlobalPtr)
      : MDeviceGlobalPtr(DeviceGlobalPtr), MDeviceGlobalTSize(0),
        MIsDeviceImageScopeDecorated(false) {}

  void initialize(std::uint32_t DeviceGlobalTSize,
                  bool IsDeviceImageScopeDecorated) {
    assert(DeviceGlobalTSize != 0 && "Device global initialized with 0 size.");
    assert(MDeviceGlobalTSize == 0 &&
           "Device global has already been initialized.");
    MDeviceGlobalTSize = DeviceGlobalTSize;
    MIsDeviceImageScopeDecorated = IsDeviceImageScopeDecorated;
  }
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
