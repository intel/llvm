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
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

DeviceGlobalUSMMem::~DeviceGlobalUSMMem() {
  // removeAssociatedResources is expected to have cleaned up both the pointer
  // and the event. When asserts are enabled the values are set, so we check
  // these here.
  assert(MPtr == nullptr && "MPtr has not been cleaned up.");
  assert(!MZeroInitEvent.has_value() &&
         "MZeroInitEvent has not been cleaned up.");
}

OwnedPiEvent DeviceGlobalUSMMem::getZeroInitEvent(const plugin &Plugin) {
  std::lock_guard<std::mutex> Lock(MZeroInitEventMutex);
  // If there is a zero-init event we can remove it if it is done.
  if (MZeroInitEvent.has_value()) {
    if (get_event_info<info::event::command_execution_status>(
            *MZeroInitEvent, Plugin) == info::event_command_status::complete) {
      Plugin.call<PiApiKind::piEventRelease>(*MZeroInitEvent);
      MZeroInitEvent = {};
      return OwnedPiEvent(Plugin);
    } else {
      return OwnedPiEvent(*MZeroInitEvent, Plugin);
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

  // Zero-initialize here and save the event.
  {
    std::lock_guard<std::mutex> Lock(NewAlloc.MZeroInitEventMutex);
    RT::PiEvent InitEvent;
    MemoryManager::fill_usm(NewAlloc.MPtr, QueueImpl, MDeviceGlobalTSize, 0,
                            std::vector<RT::PiEvent>{}, &InitEvent);
    NewAlloc.MZeroInitEvent = InitEvent;
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
      if (USMMem.MZeroInitEvent.has_value())
        CtxImpl->getPlugin().call<PiApiKind::piEventRelease>(
            *USMMem.MZeroInitEvent);
#ifndef NDEBUG
      // For debugging we set the event and memory to some recognizable values
      // to allow us to check that this cleanup happens before erasure.
      USMMem.MPtr = nullptr;
      USMMem.MZeroInitEvent = {};
#endif
      MDeviceToUSMPtrMap.erase(USMPtrIt);
    }
  }
}

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
