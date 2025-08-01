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
#include <optional>
#include <set>
#include <unordered_set>

#include <detail/ur_utils.hpp>
#include <sycl/detail/defines_elementary.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

// Forward declaration
class context_impl;
class device_impl;
class platform_impl;
class queue_impl;
class event_impl;
using EventImplPtr = std::shared_ptr<sycl::detail::event_impl>;

struct DeviceGlobalUSMMem {
  DeviceGlobalUSMMem(void *Ptr) : MPtr(Ptr) {}
  ~DeviceGlobalUSMMem();

  void *const &getPtr() const noexcept { return MPtr; }

  // Gets the initialization event if it exists. If not the OwnedUrEvent
  // will contain no event.
  OwnedUrEvent getInitEvent(adapter_impl &Adapter);

private:
  void *MPtr;
  std::mutex MInitEventMutex;
  ur_event_handle_t MInitEvent = nullptr;

  friend struct DeviceGlobalMapEntry;
};

struct DeviceGlobalMapEntry {
  // The unique identifier of the device_global.
  std::string MUniqueId;
  // Pointer to the device_global on host.
  const void *MDeviceGlobalPtr = nullptr;
  // Images device_global are used by.
  std::unordered_set<const RTDeviceBinaryImage *> MImages;
  // The image identifiers for the images using the device_global used by in the
  // cache.
  std::set<std::uintptr_t> MImageIdentifiers;
  // Size of the underlying type in the device_global.
  std::uint32_t MDeviceGlobalTSize = 0;
  // True if the device_global has been decorated with device_image_scope.
  bool MIsDeviceImageScopeDecorated = false;

  // Constructor for only initializing ID and pointer. The other members will
  // be initialized later.
  DeviceGlobalMapEntry(std::string UniqueId, const void *DeviceGlobalPtr)
      : MUniqueId(UniqueId), MDeviceGlobalPtr(DeviceGlobalPtr) {}

  // Constructor for only initializing ID, type size, and device image scope
  // flag. The pointer to the device global will be initialized later.
  DeviceGlobalMapEntry(std::string UniqueId, const RTDeviceBinaryImage *Img,
                       std::uint32_t DeviceGlobalTSize,
                       bool IsDeviceImageScopeDecorated)
      : MUniqueId(UniqueId), MImages{Img},
        MImageIdentifiers{reinterpret_cast<uintptr_t>(Img)},
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
  void initialize(const RTDeviceBinaryImage *Img,
                  std::uint32_t DeviceGlobalTSize,
                  bool IsDeviceImageScopeDecorated) {
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
    MImages.insert(Img);
    MImageIdentifiers.insert(reinterpret_cast<uintptr_t>(Img));
    MDeviceGlobalTSize = DeviceGlobalTSize;
    MIsDeviceImageScopeDecorated = IsDeviceImageScopeDecorated;
  }

  // Gets or allocates USM memory for a device_global.
  DeviceGlobalUSMMem &getOrAllocateDeviceGlobalUSM(queue_impl &QueueImpl);

  // This overload allows the allocation to be initialized without a queue. The
  // UR adapter in use must report true for
  // UR_DEVICE_INFO_USM_CONTEXT_MEMCPY_SUPPORT_EXP to take advantage of this.
  DeviceGlobalUSMMem &getOrAllocateDeviceGlobalUSM(const context &Context);

  // Removes resources for device_globals associated with the context.
  void removeAssociatedResources(const context_impl *CtxImpl);

  // Cleans up the USM memory and intialization events associated with this
  // entry. This should only be called when the device global entry is not
  // owned by the program manager, as otherwise it will be bound to the lifetime
  // of the owner context and will be cleaned up through
  // removeAssociatedResources.
  void cleanup();

private:
  // Map from a device and a context to the associated USM allocation for the
  // device_global. This should always be empty if MIsDeviceImageScopeDecorated
  // is true.
  std::map<std::pair<const device_impl *, const context_impl *>,
           DeviceGlobalUSMMem>
      MDeviceToUSMPtrMap;
  std::mutex MDeviceToUSMPtrMapMutex;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
