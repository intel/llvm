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
#include <sycl/detail/common.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

DeviceGlobalUSMMem::~DeviceGlobalUSMMem() {
  // removeAssociatedResources is expected to have cleaned up both the pointer
  // and the event. When asserts are enabled the values are set, so we check
  // these here.
  try {
    auto ContextImplPtr = MAllocatingContext.lock();
    if (ContextImplPtr) {
      if (MPtr != nullptr) {
        detail::usm::freeInternal(MPtr, ContextImplPtr.get());
        MPtr = nullptr;
      }
      if (MInitEvent != nullptr) {
        ContextImplPtr->getAdapter().call<UrApiKind::urEventRelease>(
            MInitEvent);
        MInitEvent = nullptr;
      }
    }
  } catch (std::exception &e) {
    __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~DeviceGlobalUSMMem", e);
  }
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

bool DeviceGlobalMapEntry::isAvailableInContext(
    const context_impl *CtxImpl) const {
  std::lock_guard<std::mutex> Lock{MDeviceToUSMPtrMapMutex};
  return std::any_of(
      MDeviceToUSMPtrMap.begin(), MDeviceToUSMPtrMap.end(),
      [CtxImpl](const auto &It) { return It.first.second == CtxImpl; });
}

bool DeviceGlobalMapEntry::isProfileCounter() const {
  constexpr std::string_view CounterPrefix = "__profc_";
  return std::string_view{MUniqueId}.substr(0, CounterPrefix.size()) ==
         CounterPrefix;
}

// __sycl_increment_profile_counters must be defined as a weak symbol so that
// the program will link even if the profiling runtime is not linked in. When
// compiling with MSVC there is no weak attribute, so we use a pragma comment
// and default function to achieve the same effect. When compiling with Apple
// Clang, profiling is unsupported and the function definition is empty.
#ifdef _MSC_VER
extern "C" void
__sycl_increment_profile_counters(std::uint64_t FnHash, std::size_t NumCounters,
                                  const std::uint64_t *Increments);
extern "C" void
__sycl_increment_profile_counters_default(std::uint64_t FnHash,
                                          std::size_t NumCounters,
                                          const std::uint64_t *Increments) {
  (void)FnHash;
  (void)NumCounters;
  (void)Increments;
}
#pragma comment(                                                               \
    linker,                                                                    \
    "/alternatename:__sycl_increment_profile_counters=__sycl_increment_profile_counters_default")
#elif defined(__clang__) && defined(__apple_build_version__)
extern "C" void
__sycl_increment_profile_counters(std::uint64_t FnHash, std::size_t NumCounters,
                                  const std::uint64_t *Increments) {
  (void)FnHash;
  (void)NumCounters;
  (void)Increments;
}
#else
extern "C" void __attribute__((weak))
__sycl_increment_profile_counters(std::uint64_t FnHash, std::size_t NumCounters,
                                  const std::uint64_t *Increments);
#endif

void DeviceGlobalMapEntry::cleanupProfileCounter(context_impl *CtxImpl) {
  std::lock_guard<std::mutex> Lock{MDeviceToUSMPtrMapMutex};
  assert(isProfileCounter() && "Not a profile counter device global.");
  const std::size_t NumCounters = MDeviceGlobalTSize / sizeof(std::uint64_t);
  const std::uint64_t FnHash = [&] {
    constexpr size_t PrefixSize = std::string_view{"__profc_"}.size();
    constexpr int DecimalBase = 10;
    return std::strtoull(MUniqueId.substr(PrefixSize).c_str(), nullptr,
                         DecimalBase);
  }();
  for (const device_impl &Device : CtxImpl->getDevices()) {
    auto USMPtrIt = MDeviceToUSMPtrMap.find({&Device, CtxImpl});
    if (USMPtrIt == MDeviceToUSMPtrMap.end())
      continue;

    // Get the increments from the USM pointer.
    DeviceGlobalUSMMem &USMMem = USMPtrIt->second;
    std::vector<std::uint64_t> Increments(NumCounters);
    const std::uint64_t *Counters = static_cast<std::uint64_t *>(USMMem.MPtr);
    for (std::size_t I = 0; I < NumCounters; ++I)
      Increments[I] = Counters[I];

    // Call the weak symbol to update the profile counters.
    if (&__sycl_increment_profile_counters)
      __sycl_increment_profile_counters(FnHash, Increments.size(),
                                        Increments.data());

    // Free the USM memory and release the event if it exists.
    detail::usm::freeInternal(USMMem.MPtr, CtxImpl);
    if (USMMem.MInitEvent != nullptr)
      CtxImpl->getAdapter().call<UrApiKind::urEventRelease>(USMMem.MInitEvent);

    // Set to nullptr to avoid double free.
    USMMem.MPtr = nullptr;
    USMMem.MInitEvent = nullptr;
    MDeviceToUSMPtrMap.erase(USMPtrIt);
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
      0, MDeviceGlobalTSize, &CtxImpl, &DevImpl,
      isProfileCounter() ? sycl::usm::alloc::shared : sycl::usm::alloc::device);

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
    ur_event_handle_t InitEvent = nullptr;
    if (MDeviceGlobalPtr) {
      // C++ guarantees members appear in memory in the order they are
      // declared, so since the member variable that contains the initial
      // contents of the device_global is right after the usm_ptr member
      // variable we can do some pointer arithmetic to memcopy over this
      // value to the usm_ptr. This value inside of the device_global will
      // be zero-initialized if it was not given a value on construction.
      MemoryManager::copy_usm(
          reinterpret_cast<const void *>(
              reinterpret_cast<uintptr_t>(MDeviceGlobalPtr) +
              sizeof(MDeviceGlobalPtr)),
          QueueImpl, MDeviceGlobalTSize, NewAlloc.MPtr,
          std::vector<ur_event_handle_t>{}, &InitEvent);
    } else {
      // For SYCLBIN device globals we do not have a host pointer to copy
      // from, so instead we fill the USM memory with 0's.
      MemoryManager::fill_usm(NewAlloc.MPtr, QueueImpl, MDeviceGlobalTSize,
                              {static_cast<unsigned char>(0)}, {}, &InitEvent);
    }
    NewAlloc.MInitEvent = InitEvent;
  }

  // Only device globals with host variables need to be registered with the
  // context. The rest will be managed by their kernel bundles and cleaned
  // up accordingly.
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
      0, MDeviceGlobalTSize, &CtxImpl, &DevImpl,
      isProfileCounter() ? sycl::usm::alloc::shared : sycl::usm::alloc::device);

  auto NewAllocIt = MDeviceToUSMPtrMap.emplace(
      std::piecewise_construct, std::forward_as_tuple(&DevImpl, &CtxImpl),
      std::forward_as_tuple(NewDGUSMPtr));
  assert(NewAllocIt.second &&
         "USM allocation for device and context already happened.");
  DeviceGlobalUSMMem &NewAlloc = NewAllocIt.first->second;
  NewAlloc.MAllocatingContext = CtxImpl.shared_from_this();

  if (MDeviceGlobalPtr) {
    // C++ guarantees members appear in memory in the order they are
    // declared, so since the member variable that contains the initial
    // contents of the device_global is right after the usm_ptr member
    // variable we can do some pointer arithmetic to memcopy over this value
    // to the usm_ptr. This value inside of the device_global will be
    // zero-initialized if it was not given a value on construction.
    MemoryManager::context_copy_usm(
        reinterpret_cast<const void *>(
            reinterpret_cast<uintptr_t>(MDeviceGlobalPtr) +
            sizeof(MDeviceGlobalPtr)),
        &CtxImpl, MDeviceGlobalTSize, NewAlloc.MPtr);
  } else {
    // For SYCLBIN device globals we do not have a host pointer to copy
    // from, so instead we fill the USM memory with 0's.
    std::vector<unsigned char> ImmBuff(MDeviceGlobalTSize,
                                       static_cast<unsigned char>(0));
    MemoryManager::context_copy_usm(ImmBuff.data(), &CtxImpl,
                                    MDeviceGlobalTSize, NewAlloc.MPtr);
  }

  // Only device globals with host variables need to be registered with the
  // context. The rest will be managed by their kernel bundles and cleaned
  // up accordingly.
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
