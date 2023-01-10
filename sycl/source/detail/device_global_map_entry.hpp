//==----------------- device_global_map_entry.hpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cassert>
#include <cstdint>
#include <map>
#include <mutex>

#include <sycl/detail/defines_elementary.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

// Forward declaration
class device_impl;
class context_impl;

struct DeviceGlobalMapEntry {
  // The unique identifier of the device_global.
  std::string MUniqueId;
  // Pointer to the device_global on host.
  const void *MDeviceGlobalPtr;
  // Size of the underlying type in the device_global.
  std::uint32_t MDeviceGlobalTSize;
  // True if the device_global has been decorated with device_image_scope
  bool MIsDeviceImageScopeDecorated;

  // Constructor for only initializing ID and pointer. The other members will
  // be initialized later.
  DeviceGlobalMapEntry(std::string UniqueId, const void *DeviceGlobalPtr)
      : MUniqueId(UniqueId), MDeviceGlobalPtr(DeviceGlobalPtr),
        MDeviceGlobalTSize(0), MIsDeviceImageScopeDecorated(false) {}

  // Constructor for only initializing ID, type size, and device image scope
  // flag. The pointer to the device global will be initialized later.
  DeviceGlobalMapEntry(std::string UniqueId, std::uint32_t DeviceGlobalTSize,
                       bool IsDeviceImageScopeDecorated)
      : MUniqueId(UniqueId), MDeviceGlobalPtr(nullptr),
        MDeviceGlobalTSize(DeviceGlobalTSize),
        MIsDeviceImageScopeDecorated(IsDeviceImageScopeDecorated) {}

  // Initialize the pointer to the associated device_global.
  void initialize(const void *DeviceGlobalPtr) {
    assert(DeviceGlobalPtr && "Device global pointer cannot be null");
    assert(!MDeviceGlobalPtr &&
           "Device global pointer has already been initialized.");
    MDeviceGlobalPtr = DeviceGlobalPtr;
  }

  // Initialize the device_global's element type size and the flag signalling
  // if the device_global has the device_image_scope property.
  void initialize(std::uint32_t DeviceGlobalTSize,
                  bool IsDeviceImageScopeDecorated) {
    assert(DeviceGlobalTSize != 0 && "Device global initialized with 0 size.");
    if (MDeviceGlobalTSize != 0) {
      // The device global entry has already been initialized. This can happen
      // if multiple images contain the device-global. They must agree on the
      // information.
      assert(MDeviceGlobalTSize == DeviceGlobalTSize &&
             "Device global intializations disagree on type size.");
      assert(
          MIsDeviceImageScopeDecorated == IsDeviceImageScopeDecorated &&
          "Device global intializations disagree on image scope decoration.");
      return;
    }
    MDeviceGlobalTSize = DeviceGlobalTSize;
    MIsDeviceImageScopeDecorated = IsDeviceImageScopeDecorated;
  }

private:
  // Map from a device and a context to the associated USM allocation for the
  // device_global. This should always be empty if MIsDeviceImageScopeDecorated
  // is true.
  std::map<std::pair<const device_impl *, const context_impl *>, void *>
      MDeviceToUSMPtrMap;
  std::mutex MDeviceToUSMPtrMapMutex;
};

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
