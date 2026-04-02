//==-------------------- device_global_map.hpp -----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mutex>
#include <string_view>
#include <unordered_map>

#include <detail/compiler.hpp>
#include <detail/device_binary_image.hpp>
#include <detail/device_global_map_entry.hpp>
#include <sycl/detail/defines_elementary.hpp>
#include <sycl/kernel_bundle.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

class DeviceGlobalMap {
public:
  DeviceGlobalMap(bool OwnerControlledCleanup)
      : MOwnerControlledCleanup{OwnerControlledCleanup} {}

  DeviceGlobalMap(const DeviceGlobalMap &) = delete;
  DeviceGlobalMap &operator=(const DeviceGlobalMap &) = delete;

  ~DeviceGlobalMap() {
    try {
      if (!MOwnerControlledCleanup)
        for (auto &DeviceGlobalIt : MDeviceGlobals)
          DeviceGlobalIt.second->cleanup();
    } catch (std::exception &e) {
      __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~DeviceGlobalMap", e);
    }
  }

  void initializeEntries(const RTDeviceBinaryImage *Img) {
    std::lock_guard<std::mutex> DeviceGlobalsGuard(MDeviceGlobalsMutex);
    initializeEntriesLockless(Img);
  }

  void initializeEntriesLockless(const RTDeviceBinaryImage *Img) {
    const auto &DeviceGlobals = Img->getDeviceGlobals();
    for (const sycl_device_binary_property &DeviceGlobal : DeviceGlobals) {
      ByteArray DeviceGlobalInfo =
          DeviceBinaryProperty(DeviceGlobal).asByteArray();

      // The supplied device_global info property is expected to contain:
      // * 8 bytes - Size of the property.
      // * 4 bytes - Size of the underlying type in the device_global.
      // * 4 bytes - 0 if device_global has device_image_scope and any value
      //             otherwise.
      DeviceGlobalInfo.dropBytes(8);
      auto [TypeSize, DeviceImageScopeDecorated] =
          DeviceGlobalInfo.consume<std::uint32_t, std::uint32_t>();
      assert(DeviceGlobalInfo.empty() && "Extra data left!");

      // Give the image pointer as an identifier for the image the
      // device-global is associated with.

      auto ExistingDeviceGlobal = MDeviceGlobals.find(DeviceGlobal->Name);
      if (ExistingDeviceGlobal != MDeviceGlobals.end()) {
        // If it has already been registered we update the information.
        ExistingDeviceGlobal->second->initialize(Img, TypeSize,
                                                 DeviceImageScopeDecorated);
      } else {
        // If it has not already been registered we create a new entry.
        // Note: Pointer to the device global is not available here, so it
        //       cannot be set until registration happens.
        auto EntryUPtr = std::make_unique<DeviceGlobalMapEntry>(
            DeviceGlobal->Name, Img, TypeSize, DeviceImageScopeDecorated);
        auto NewEntry =
            MDeviceGlobals.emplace(DeviceGlobal->Name, std::move(EntryUPtr));
        if (NewEntry.first->second->isProfileCounter())
          MProfileCounterDeviceGlobals.push_back(NewEntry.first->second.get());
      }
    }
  }

  void eraseEntries(const RTDeviceBinaryImage *Img) {
    const auto &DeviceGlobals = Img->getDeviceGlobals();
    std::lock_guard<std::mutex> DeviceGlobalsGuard(MDeviceGlobalsMutex);
    for (const sycl_device_binary_property &DeviceGlobal : DeviceGlobals) {
      if (auto DevGlobalIt = MDeviceGlobals.find(DeviceGlobal->Name);
          DevGlobalIt != MDeviceGlobals.end()) {
        auto findDevGlobalByValue = std::find_if(
            MPtr2DeviceGlobal.begin(), MPtr2DeviceGlobal.end(),
            [&DevGlobalIt](
                const std::pair<const void *, DeviceGlobalMapEntry *> &Entry) {
              return Entry.second == DevGlobalIt->second.get();
            });
        if (findDevGlobalByValue != MPtr2DeviceGlobal.end())
          MPtr2DeviceGlobal.erase(findDevGlobalByValue);

        MDeviceGlobals.erase(DevGlobalIt);
      }
    }
  }

  void addOrInitialize(const void *DeviceGlobalPtr, const char *UniqueId) {
    std::lock_guard<std::mutex> DeviceGlobalsGuard(MDeviceGlobalsMutex);
    auto ExistingDeviceGlobal = MDeviceGlobals.find(UniqueId);
    if (ExistingDeviceGlobal != MDeviceGlobals.end()) {
      // Update the existing information and add the entry to the pointer map.
      ExistingDeviceGlobal->second->initialize(DeviceGlobalPtr);
      MPtr2DeviceGlobal.insert(
          {DeviceGlobalPtr, ExistingDeviceGlobal->second.get()});
      return;
    }

    auto EntryUPtr =
        std::make_unique<DeviceGlobalMapEntry>(UniqueId, DeviceGlobalPtr);
    auto NewEntry = MDeviceGlobals.emplace(UniqueId, std::move(EntryUPtr));
    if (NewEntry.first->second->isProfileCounter())
      MProfileCounterDeviceGlobals.push_back(NewEntry.first->second.get());
    MPtr2DeviceGlobal.insert({DeviceGlobalPtr, NewEntry.first->second.get()});
  }

  DeviceGlobalMapEntry *getEntry(const void *DeviceGlobalPtr) {
    std::lock_guard<std::mutex> DeviceGlobalsGuard(MDeviceGlobalsMutex);
    auto Entry = MPtr2DeviceGlobal.find(DeviceGlobalPtr);
    return (Entry != MPtr2DeviceGlobal.end()) ? Entry->second : nullptr;
  }

  DeviceGlobalMapEntry *
  tryGetEntry(const std::string &UniqueId,
              bool ExcludeDeviceImageScopeDecorated = false) {
    std::lock_guard<std::mutex> DeviceGlobalsGuard(MDeviceGlobalsMutex);
    return tryGetEntryLockless(UniqueId, ExcludeDeviceImageScopeDecorated);
  }

  DeviceGlobalMapEntry *
  tryGetEntryLockless(const std::string &UniqueId,
                      bool ExcludeDeviceImageScopeDecorated = false) const {
    auto DeviceGlobalEntry = MDeviceGlobals.find(UniqueId);
    if (DeviceGlobalEntry != MDeviceGlobals.end() &&
        (!ExcludeDeviceImageScopeDecorated ||
         !DeviceGlobalEntry->second->MIsDeviceImageScopeDecorated))
      return DeviceGlobalEntry->second.get();
    return nullptr;
  }

  void getEntries(const std::vector<std::string> &UniqueIds,
                  bool ExcludeDeviceImageScopeDecorated,
                  std::vector<DeviceGlobalMapEntry *> &OutVec) {
    std::lock_guard<std::mutex> DeviceGlobalsGuard(MDeviceGlobalsMutex);
    for (const std::string &UniqueId : UniqueIds) {
      auto DeviceGlobalEntry = MDeviceGlobals.find(UniqueId);
      if (DeviceGlobalEntry != MDeviceGlobals.end() &&
          (!ExcludeDeviceImageScopeDecorated ||
           !DeviceGlobalEntry->second->MIsDeviceImageScopeDecorated))
        OutVec.push_back(DeviceGlobalEntry->second.get());
    }
  }

  std::vector<DeviceGlobalMapEntry *> getProfileCounterEntries() {
    std::lock_guard<std::mutex> DeviceGlobalsGuard(MDeviceGlobalsMutex);
    return MProfileCounterDeviceGlobals;
  }

  const std::unordered_map<const void *, DeviceGlobalMapEntry *>
  getPointerMap() const {
    return MPtr2DeviceGlobal;
  }

  size_t size() const { return MDeviceGlobals.size(); }

  size_t count(std::string_view UniqueId) const {
    return MDeviceGlobals.count(UniqueId);
  }

private:
  // Indicates whether the owner will explicitly cleanup the entries. If false
  // the dtor of DeviceGlobalMap will cleanup the entries.
  // Note: This lets the global device global map avoid overhead at shutdown and
  //       instead let the contexts own the associated entries.
  bool MOwnerControlledCleanup = true;

  // Maps between device_global identifiers and associated information.
  std::unordered_map<std::string_view, std::unique_ptr<DeviceGlobalMapEntry>>
      MDeviceGlobals;
  std::unordered_map<const void *, DeviceGlobalMapEntry *> MPtr2DeviceGlobal;

  // List of profile counter device globals.
  std::vector<DeviceGlobalMapEntry *> MProfileCounterDeviceGlobals;

  /// Protects MDeviceGlobals and MPtr2DeviceGlobal.
  std::mutex MDeviceGlobalsMutex;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
