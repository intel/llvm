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
  auto ContextImplPtr = MAllocatingContext.lock();
  if (ContextImplPtr) {
    if (MPtr != nullptr) {
      detail::usm::freeInternal(MPtr, ContextImplPtr.get());
      MPtr = nullptr;
    }
    if (MInitEvent != nullptr) {
      ContextImplPtr->getAdapter().call<UrApiKind::urEventRelease>(MInitEvent);
      MInitEvent = nullptr;
    }
  }

  assert(MPtr == nullptr && "MPtr has not been cleaned up.");
  assert(MInitEvent == nullptr && "MInitEvent has not been cleaned up.");
}

OwnedUrEvent DeviceGlobalUSMMem::getInitEvent(adapter_impl &Adapter) {
  std::lock_guard<std::mutex> Lock(MInitEventMutex);
  if (MInitEvent == nullptr)
    return OwnedUrEvent(Adapter);

  // If there is a init event we can remove it if it is done.
  if (get_event_info<info::event::command_execution_status>(
          MInitEvent, Adapter) == info::event_command_status::complete) {
    Adapter.call<UrApiKind::urEventRelease>(MInitEvent);
    MInitEvent = nullptr;
    return OwnedUrEvent(Adapter);
  } else {
    return OwnedUrEvent(MInitEvent, Adapter);
  }
}

DeviceGlobalUSMMem &
DeviceGlobalMapEntry::getOrAllocateDeviceGlobalUSM(queue_impl &QueueImpl) {
  assert(!MIsDeviceImageScopeDecorated &&
         "USM allocations should not be acquired for device_global with "
         "device_image_scope property.");
  context_impl &CtxImpl = QueueImpl.getContextImpl();
  const device_impl &DevImpl = QueueImpl.getDeviceImpl();
  std::lock_guard<std::mutex> Lock(MDeviceToUSMPtrMapMutex);

  auto DGUSMPtr = MDeviceToUSMPtrMap.find({&DevImpl, &CtxImpl});
  if (DGUSMPtr != MDeviceToUSMPtrMap.end())
    return DGUSMPtr->second;

  void *NewDGUSMPtr = detail::usm::alignedAllocInternal(
      0, MDeviceGlobalTSize, &CtxImpl, &DevImpl, sycl::usm::alloc::device);

  auto NewAllocIt = MDeviceToUSMPtrMap.emplace(
      std::piecewise_construct, std::forward_as_tuple(&DevImpl, &CtxImpl),
      std::forward_as_tuple(NewDGUSMPtr));
  assert(NewAllocIt.second &&
         "USM allocation for device and context already happened.");
  DeviceGlobalUSMMem &NewAlloc = NewAllocIt.first->second;
  NewAlloc.MAllocatingContext = CtxImpl.shared_from_this();

  // Initialize here and save the event.
  {
    std::lock_guard<std::mutex> Lock(NewAlloc.MInitEventMutex);
    ur_event_handle_t InitEvent;
    if (MDeviceGlobalPtr) {
      // C++ guarantees members appear in memory in the order they are declared,
      // so since the member variable that contains the initial contents of the
      // device_global is right after the usm_ptr member variable we can do
      // some pointer arithmetic to memcopy over this value to the usm_ptr. This
      // value inside of the device_global will be zero-initialized if it was
      // not given a value on construction.
      MemoryManager::copy_usm(
          reinterpret_cast<const void *>(
              reinterpret_cast<uintptr_t>(MDeviceGlobalPtr) +
              sizeof(MDeviceGlobalPtr)),
          QueueImpl, MDeviceGlobalTSize, NewAlloc.MPtr,
          std::vector<ur_event_handle_t>{}, &InitEvent);
    } else {
      // For SYCLBIN device globals we do not have a host pointer to copy from,
      // so instead we fill the USM memory with 0's.
      MemoryManager::fill_usm(NewAlloc.MPtr, QueueImpl, MDeviceGlobalTSize,
                              {static_cast<unsigned char>(0)}, {}, &InitEvent);
    }
    NewAlloc.MInitEvent = InitEvent;
  }

  // Only device globals with host variables need to be registered with the
  // context. The rest will be managed by their kernel bundles and cleaned up
  // accordingly.
  if (MDeviceGlobalPtr)
    CtxImpl.addAssociatedDeviceGlobal(MDeviceGlobalPtr);
  return NewAlloc;
}

DeviceGlobalUSMMem &
DeviceGlobalMapEntry::getOrAllocateDeviceGlobalUSM(const context &Context) {
  assert(!MIsDeviceImageScopeDecorated &&
         "USM allocations should not be acquired for device_global with "
         "device_image_scope property.");
  context_impl &CtxImpl = *getSyclObjImpl(Context);
  device_impl &DevImpl = CtxImpl.getDevices().front();
  std::lock_guard<std::mutex> Lock(MDeviceToUSMPtrMapMutex);

  auto DGUSMPtr = MDeviceToUSMPtrMap.find({&DevImpl, &CtxImpl});
  if (DGUSMPtr != MDeviceToUSMPtrMap.end())
    return DGUSMPtr->second;

  void *NewDGUSMPtr = detail::usm::alignedAllocInternal(
      0, MDeviceGlobalTSize, &CtxImpl, &DevImpl, sycl::usm::alloc::device);

  auto NewAllocIt = MDeviceToUSMPtrMap.emplace(
      std::piecewise_construct, std::forward_as_tuple(&DevImpl, &CtxImpl),
      std::forward_as_tuple(NewDGUSMPtr));
  assert(NewAllocIt.second &&
         "USM allocation for device and context already happened.");
  DeviceGlobalUSMMem &NewAlloc = NewAllocIt.first->second;
  NewAlloc.MAllocatingContext = CtxImpl.shared_from_this();

  if (MDeviceGlobalPtr) {
    // C++ guarantees members appear in memory in the order they are declared,
    // so since the member variable that contains the initial contents of the
    // device_global is right after the usm_ptr member variable we can do
    // some pointer arithmetic to memcopy over this value to the usm_ptr. This
    // value inside of the device_global will be zero-initialized if it was not
    // given a value on construction.
    MemoryManager::context_copy_usm(
        reinterpret_cast<const void *>(
            reinterpret_cast<uintptr_t>(MDeviceGlobalPtr) +
            sizeof(MDeviceGlobalPtr)),
        &CtxImpl, MDeviceGlobalTSize, NewAlloc.MPtr);
  } else {
    // For SYCLBIN device globals we do not have a host pointer to copy from,
    // so instead we fill the USM memory with 0's.
    std::vector<unsigned char> ImmBuff(MDeviceGlobalTSize,
                                       static_cast<unsigned char>(0));
    MemoryManager::context_copy_usm(ImmBuff.data(), &CtxImpl,
                                    MDeviceGlobalTSize, NewAlloc.MPtr);
  }

  // Only device globals with host variables need to be registered with the
  // context. The rest will be managed by their kernel bundles and cleaned up
  // accordingly.
  if (MDeviceGlobalPtr)
    CtxImpl.addAssociatedDeviceGlobal(MDeviceGlobalPtr);
  return NewAlloc;
}

void DeviceGlobalMapEntry::removeAssociatedResources(
    const context_impl *CtxImpl) {
  std::lock_guard<std::mutex> Lock{MDeviceToUSMPtrMapMutex};
  for (device_impl &Device : CtxImpl->getDevices()) {
    auto USMPtrIt = MDeviceToUSMPtrMap.find({&Device, CtxImpl});
    if (USMPtrIt != MDeviceToUSMPtrMap.end()) {
      DeviceGlobalUSMMem &USMMem = USMPtrIt->second;
      detail::usm::freeInternal(USMMem.MPtr, CtxImpl);
      if (USMMem.MInitEvent != nullptr)
        CtxImpl->getAdapter().call<UrApiKind::urEventRelease>(
            USMMem.MInitEvent);
      // Set to nullptr to avoid double free.
      USMMem.MPtr = nullptr;
      USMMem.MInitEvent = nullptr;
      MDeviceToUSMPtrMap.erase(USMPtrIt);
    }
  }
}

void DeviceGlobalMapEntry::cleanup() {
  std::lock_guard<std::mutex> Lock{MDeviceToUSMPtrMapMutex};
  assert(MDeviceGlobalPtr == nullptr &&
         "Entry has host variable, so it should be associated with a context "
         "and should be cleaned up by its dtor.");
  for (auto &USMPtrIt : MDeviceToUSMPtrMap) {
    // The context should be alive through the kernel_bundle owning these
    // device_global entries.
    const context_impl *CtxImpl = USMPtrIt.first.second;
    DeviceGlobalUSMMem &USMMem = USMPtrIt.second;
    detail::usm::freeInternal(USMMem.MPtr, CtxImpl);
    if (USMMem.MInitEvent != nullptr)
      CtxImpl->getAdapter().call<UrApiKind::urEventRelease>(USMMem.MInitEvent);
    // Set to nullptr to avoid double free.
    USMMem.MPtr = nullptr;
    USMMem.MInitEvent = nullptr;
  }
  MDeviceToUSMPtrMap.clear();
}

} // namespace detail
} // namespace _V1
} // namespace sycl
