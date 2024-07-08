//==------------------ device_global_map_entry.cpp -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/context_impl.hpp>
#include <detail/device_global_map_entry.hpp>
#include <detail/event_info.hpp>
#include <detail/memory_manager.hpp>
#include <detail/queue_impl.hpp>
#include <detail/usm/usm_impl.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

DeviceGlobalUSMMem::~DeviceGlobalUSMMem() {
  // removeAssociatedResources is expected to have cleaned up both the pointer
  // and the event. When asserts are enabled the values are set, so we check
  // these here.
  assert(MPtr == nullptr && "MPtr has not been cleaned up.");
  assert(!MInitEvent.has_value() && "MInitEvent has not been cleaned up.");
}

OwnedPiEvent DeviceGlobalUSMMem::getInitEvent(const PluginPtr &Plugin) {
  std::lock_guard<std::mutex> Lock(MInitEventMutex);
  // If there is a init event we can remove it if it is done.
  if (MInitEvent.has_value()) {
    if (get_event_info<info::event::command_execution_status>(
            *MInitEvent, Plugin) == info::event_command_status::complete) {
      Plugin->call<PiApiKind::piEventRelease>(*MInitEvent);
      MInitEvent = {};
      return OwnedPiEvent(Plugin);
    } else {
      return OwnedPiEvent(*MInitEvent, Plugin);
    }
  }
  return OwnedPiEvent(Plugin);
}

DeviceGlobalUSMMem &DeviceGlobalMapEntry::getOrAllocateDeviceGlobalUSM(
    const std::shared_ptr<queue_impl> &QueueImpl) {
  assert(!MIsDeviceImageScopeDecorated &&
         "USM allocations should not be acquired for device_global with "
         "device_image_scope property.");
  const std::shared_ptr<context_impl> &CtxImpl = QueueImpl->getContextImplPtr();
  const std::shared_ptr<device_impl> &DevImpl = QueueImpl->getDeviceImplPtr();
  std::lock_guard<std::mutex> Lock(MDeviceToUSMPtrMapMutex);

  auto DGUSMPtr = MDeviceToUSMPtrMap.find({DevImpl.get(), CtxImpl.get()});
  if (DGUSMPtr != MDeviceToUSMPtrMap.end())
    return DGUSMPtr->second;

  void *NewDGUSMPtr = detail::usm::alignedAllocInternal(
      0, MDeviceGlobalTSize, CtxImpl.get(), DevImpl.get(),
      sycl::usm::alloc::device);

  auto NewAllocIt = MDeviceToUSMPtrMap.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(DevImpl.get(), CtxImpl.get()),
      std::forward_as_tuple(NewDGUSMPtr));
  assert(NewAllocIt.second &&
         "USM allocation for device and context already happened.");
  DeviceGlobalUSMMem &NewAlloc = NewAllocIt.first->second;

  // Initialize here and save the event.
  {
    std::lock_guard<std::mutex> Lock(NewAlloc.MInitEventMutex);
    sycl::detail::pi::PiEvent InitEvent;
    // C++ guarantees members appear in memory in the order they are declared,
    // so since the member variable that contains the initial contents of the
    // device_global is right after the usm_ptr member variable we can do
    // some pointer arithmetic to memcopy over this value to the usm_ptr. This
    // value inside of the device_global will be zero-initialized if it was not
    // given a value on construction.
    MemoryManager::copy_usm(reinterpret_cast<const void *>(
                                reinterpret_cast<uintptr_t>(MDeviceGlobalPtr) +
                                sizeof(MDeviceGlobalPtr)),
                            QueueImpl, MDeviceGlobalTSize, NewAlloc.MPtr,
                            std::vector<sycl::detail::pi::PiEvent>{},
                            &InitEvent, nullptr);
    NewAlloc.MInitEvent = InitEvent;
  }

  CtxImpl->addAssociatedDeviceGlobal(MDeviceGlobalPtr);
  return NewAlloc;
}

void DeviceGlobalMapEntry::removeAssociatedResources(
    const context_impl *CtxImpl) {
  std::lock_guard<std::mutex> Lock{MDeviceToUSMPtrMapMutex};
  for (device Device : CtxImpl->getDevices()) {
    auto USMPtrIt =
        MDeviceToUSMPtrMap.find({getSyclObjImpl(Device).get(), CtxImpl});
    if (USMPtrIt != MDeviceToUSMPtrMap.end()) {
      DeviceGlobalUSMMem &USMMem = USMPtrIt->second;
      detail::usm::freeInternal(USMMem.MPtr, CtxImpl);
      if (USMMem.MInitEvent.has_value())
        CtxImpl->getPlugin()->call<PiApiKind::piEventRelease>(
            *USMMem.MInitEvent);
#ifndef NDEBUG
      // For debugging we set the event and memory to some recognizable values
      // to allow us to check that this cleanup happens before erasure.
      USMMem.MPtr = nullptr;
      USMMem.MInitEvent = {};
#endif
      MDeviceToUSMPtrMap.erase(USMPtrIt);
    }
  }
}

} // namespace detail
} // namespace _V1
} // namespace sycl
