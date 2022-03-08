//==-------------- resource_pool.cpp - USM resource pool -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/resource_pool.hpp>
#include <detail/config.hpp>
#include <detail/context_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/mem_alloc_helper.hpp>
#include <detail/platform_impl.hpp>
#include <detail/queue_impl.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

ManagedResourceBase::~ManagedResourceBase() {
  // Only return it to the pool if resource pooling is enabled.
  if (MOrigin->MIsPoolingEnabled)
    MOrigin->returnResourceToPool(MSize, MMem);
}

ResourcePool::ResourcePool()
    : MIsPoolingEnabled(
          !SYCLConfig<SYCL_DISABLE_AUXILIARY_RESOURCE_POOL>::get()) {}

void ResourcePool::clear() {
  std::lock_guard<std::mutex> Lock{MMutex};
  MAllocCount -= MFreeEntries.size();
  for (auto Entry : MFreeEntries)
    memReleaseHelper(MPlatform->getPlugin(), Entry.Mem);
  MFreeEntries.clear();
}

ResourcePool::FreeEntry ResourcePool::getOrAllocateEntry(
    const size_t Size, const std::shared_ptr<context_impl> &ContextImplPtr,
    void *DataPtr, bool *IsNewEntry) {
  assert(Size && "Size must be greater than 0");

  {
    std::lock_guard<std::mutex> Lock{MMutex};

    // Store platform to allow future freeing.
    if (!MPlatform)
      MPlatform = ContextImplPtr->getPlatformImpl();

    // Find the free entry with the smallest suitable size.
    auto FoundFreeEntry = MFreeEntries.upper_bound(Size - 1);

    // If there was a fitting free entry in the pool, remove and return it.
    const bool IsOldEntry = FoundFreeEntry != MFreeEntries.end();
    if (IsNewEntry)
      *IsNewEntry = !IsOldEntry;
    if (IsOldEntry) {
      FreeEntry Entry = *FoundFreeEntry;
      MFreeEntries.erase(FoundFreeEntry);
      return Entry;
    }
  }

  // If there was no suitable free entry we allocate memory and return it in a
  // new free entry.
  RT::PiMemFlags MemFlags = PI_MEM_FLAGS_ACCESS_RW;
  if (DataPtr)
    MemFlags |= PI_MEM_FLAGS_HOST_PTR_COPY;
  RT::PiMem NewResMem;
  memBufferCreateHelper(ContextImplPtr, MemFlags, Size, DataPtr, &NewResMem,
                        nullptr);
  ++MAllocCount;
  return {Size, NewResMem};
}

ResourcePool::FreeEntry ResourcePool::getOrAllocateEntry(
    const size_t Size, const std::shared_ptr<queue_impl> &QueueImplPtr,
    void *DataPtr, event *AvailableEvent, bool *IsNewEntry) {
  ResourcePool::FreeEntry Entry = getOrAllocateEntry(
      Size, QueueImplPtr->getContextImplPtr(), DataPtr, IsNewEntry);

  // A new entry will have copied from the host pointer on creation.
  if (IsNewEntry) {
    *AvailableEvent = event{};
    return Entry;
  }

  // If we get old memory we need to copy explicitly.
  RT::PiEvent Event;
  QueueImplPtr->getPlugin().call<PiApiKind::piEnqueueMemBufferWrite>(
      QueueImplPtr->getHandleRef(), Entry.Mem,
      /*blocking_write=*/CL_FALSE, 0, Size, DataPtr, 0, nullptr, &Event);
  *AvailableEvent = createSyclObjFromImpl<event>(
      std::make_shared<event_impl>(Event, QueueImplPtr->get_context()));
  return Entry;
}

const std::shared_ptr<context_impl> &ResourcePool::getQueueContextImpl(
    const std::shared_ptr<queue_impl> &QueueImplPtr) {
  return QueueImplPtr->getContextImplPtr();
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
