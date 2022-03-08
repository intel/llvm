//==------------- resource_pool.hpp - USM resource pool ---------*- C++-*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/buffer.hpp>
#include <CL/sycl/detail/defines_elementary.hpp>

#include <cassert>
#include <memory>
#include <mutex>
#include <set>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

// Forward declarations
class context_impl;
class queue_impl;
class device_impl;
class platform_impl;
class ResourcePool;

struct __SYCL_EXPORT ManagedResourceBase {
  ManagedResourceBase() = delete;
  ~ManagedResourceBase();

protected:
  ManagedResourceBase(size_t Size, RT::PiMem Mem, ResourcePool *Origin)
      : MSize(Size), MMem(Mem), MOrigin(Origin) {}

  /// Size of the memory in the managed resource.
  size_t MSize;

  /// Memory associated with the managed resource.
  RT::PiMem MMem;

  /// The resource pool the resource was taken from.
  ResourcePool *MOrigin;

  friend class ResourcePool;
};

template <typename T, int Dims>
struct ManagedResource : public ManagedResourceBase {
  ManagedResource() = delete;

  /// Gets the buffer associated with the resource.
  ///
  /// \return the buffer associated with the resource.
  buffer<T, Dims, buffer_allocator, void> &getBuffer() { return MBuffer; }

private:
  /// Creates a buffer implementation.
  ///
  /// \param Size is the size of the memory passed to the buffer.
  /// \param Mem is the memory for the buffer.
  /// \param ContextImplPtr is the context implementation the memory is
  ///        associated with.
  /// \param AvailableEvent is an event tied to the availability of the data in
  ///        the memory.
  /// \return a shared pointer to the resulting buffer implementation.
  static std::shared_ptr<buffer_impl>
  createBufferImpl(size_t Size, RT::PiMem Mem,
                   const std::shared_ptr<context_impl> &ContextImplPtr,
                   event AvailableEvent) {
    return std::make_shared<buffer_impl>(
        Mem, createSyclObjFromImpl<context>(ContextImplPtr), Size,
        make_unique_ptr<detail::SYCLMemObjAllocatorHolder<buffer_allocator>>(),
        AvailableEvent);
  }

  ManagedResource(size_t Size, RT::PiMem Mem, ResourcePool *Origin,
                  range<Dims> Range,
                  const std::shared_ptr<context_impl> &ContextImplPtr,
                  event AvailableEvent = event{})
      : ManagedResourceBase(Size, Mem, Origin),
        MBuffer(createBufferImpl(Size, Mem, ContextImplPtr, AvailableEvent),
                Range, 0,
                /*IsSubBuffer=*/false) {}

  // Constructor for when pool is disabled.
  ManagedResource(ResourcePool *Origin, range<Dims> Range, T *DataPtr)
      : ManagedResourceBase(0, nullptr, Origin), MBuffer(DataPtr, Range) {}

  /// Buffer owned by the resource.
  buffer<T, Dims, buffer_allocator, void> MBuffer;

  friend class ResourcePool;
};

class __SYCL_EXPORT ResourcePool {
private:
  /// Free entry in the resource pool. This represents an allocation owned by
  /// the pool that is not currently in use.
  struct FreeEntry {
    /// Byte size of the free entry.
    size_t Size;
    /// Memory allocation of the free entry.
    RT::PiMem Mem;
  };

  /// Comparison of free entries by size. This is used for fast lookup by size
  /// in the pool.
  struct FreeEntryCompare {
    using is_transparent = void;
    bool operator()(FreeEntry const &lhs, FreeEntry const &rhs) const {
      return lhs.Size < rhs.Size;
    }
    bool operator()(FreeEntry const &lhs, size_t rhs) const {
      return lhs.Size < rhs;
    }
    bool operator()(size_t lhs, FreeEntry const &rhs) const {
      return lhs < rhs.Size;
    }
  };

  /// Extracts a free entry from the pool that fits the size required. If there
  /// is no suitable entry, new memory will be allocated.
  ///
  /// \param Range is the range of the resulting buffer.
  /// \param ContextImplPtr is the context to allocate memory in.
  /// \param DataPtr is the pointer to data on the host to initialize the
  ///        associated memory with. This will only be used if a new entry is
  ///        allocated.
  /// \param IsNewEntry will be set to true if the entry was newly allocated in
  ///        the pool and false if it was found in the existing free entries in
  ///        the pool. This is not set if it is nullptr.
  /// \return a shared pointer to the new managed resource.
  FreeEntry
  getOrAllocateEntry(const size_t Size,
                     const std::shared_ptr<context_impl> &ContextImplPtr,
                     void *DataPtr = nullptr, bool *IsNewEntry = nullptr);

  /// Extracts a free entry from the pool that fits the size required. If there
  /// is no suitable entry, new memory will be allocated. The memory will be
  /// initialized with the data given.
  ///
  /// \param Size is the size of the free entry to find or allocate.
  /// \param QueueImplPtr is the queue with the context to allocate memory in.
  /// \param DataPtr is the pointer to data on the host to initialize the
  ///        associated memory with.
  /// \param AvailableEvent will be set to an event that is tied to the
  ///        initialization of the memory.
  /// \param IsNewEntry will be set to true if the entry was newly allocated in
  ///        the pool and false if it was found in the existing free entries in
  ///        the pool. This is not set if it is nullptr.
  /// \return a shared pointer to the new managed resource.
  FreeEntry getOrAllocateEntry(const size_t Size,
                               const std::shared_ptr<queue_impl> &QueueImplPtr,
                               void *DataPtr, event *AvailableEvent,
                               bool *IsNewEntry = nullptr);

  /// Gets the context implementation associtated with a queue implementation.
  ///
  /// \param QueueImplPtr is the queue implementation to get the context
  /// implementation from. \return the context implementation from the queue
  /// implementation.
  static const std::shared_ptr<context_impl> &
  getQueueContextImpl(const std::shared_ptr<queue_impl> &QueueImplPtr);

  using ContextPtr = context_impl *;

public:
  /// Removes and deallocates all free entries currently in the pool.
  void clear();

  ResourcePool();
  ResourcePool(const ResourcePool &) = delete;
  ~ResourcePool() {
    clear();
    assert(MAllocCount == 0 && "Not all resources have returned to the pool.");
  }

  /// Returns true if the resource pool is enabled and false otherwise.
  ///
  /// \return a boolean value specifying whether the pool is enabled.
  bool isEnabled() { return MIsPoolingEnabled; }

  /// Creates a managed resource from the pool.
  ///
  /// \param Range is the range of the resulting buffer.
  /// \param ContextImplPtr is the context to allocate memory in.
  /// \return a shared pointer to the new managed resource.
  template <typename T, int Dims>
  std::shared_ptr<ManagedResource<T, Dims>>
  getOrAllocateResource(range<Dims> Range,
                        const std::shared_ptr<context_impl> &ContextImplPtr) {
    // If pool is disabled we return a buffer that will not return to the pool.
    if (!MIsPoolingEnabled)
      return std::shared_ptr<ManagedResource<T, Dims>>{
          new ManagedResource<T, Dims>(this, Range, nullptr)};

    // Get or allocate a free entry that fits the requirements.
    FreeEntry Entry =
        getOrAllocateEntry(Range.size() * sizeof(T), ContextImplPtr);
    return std::shared_ptr<ManagedResource<T, Dims>>{
        new ManagedResource<T, Dims>(Entry.Size, Entry.Mem, this, Range,
                                     ContextImplPtr)};
  }

  /// Creates a managed resource from the pool and sets te data of the
  /// associated memory.
  ///
  /// \param Range is the range of the resulting buffer.
  /// \param QueueImplPtr is the queue with the context to allocate memory in.
  /// \param DataPtr is the pointer to data on the host to initialize the
  /// resource with. This must contain at least the size of Range.
  /// \return a shared pointer to the new managed resource.
  template <typename T, int Dims>
  std::shared_ptr<ManagedResource<T, Dims>>
  getOrAllocateResource(range<Dims> Range,
                        const std::shared_ptr<queue_impl> &QueueImplPtr,
                        T *DataPtr) {
    // If pool is disabled we return a buffer that will not return to the pool.
    if (!MIsPoolingEnabled)
      return std::shared_ptr<ManagedResource<T, Dims>>{
          new ManagedResource<T, Dims>(this, Range, DataPtr)};

    // Get or allocate a free entry that fits the requirements.
    event AvailableEvent;
    FreeEntry Entry = getOrAllocateEntry(Range.size() * sizeof(T), QueueImplPtr,
                                         DataPtr, &AvailableEvent);
    return std::shared_ptr<ManagedResource<T, Dims>>{
        new ManagedResource<T, Dims>(Entry.Size, Entry.Mem, this, Range,
                                     getQueueContextImpl(QueueImplPtr),
                                     AvailableEvent)};
  }

private:
  /// Returns a resouce to the pool.
  ///
  /// \param Size is the size of the resource.
  /// \param Mem is the memory of the resource.
  void returnResourceToPool(const size_t Size, RT::PiMem Mem) {
    std::lock_guard<std::mutex> Lock{MMutex};
    MFreeEntries.insert({Size, Mem});
  }

  friend struct ManagedResourceBase;

  /// Is true if the pool is enabled and false otherwise. This is controlled by
  /// the SYCL_DISABLE_AUXILIARY_RESOURCE_POOL config.
  const bool MIsPoolingEnabled;

  /// The platform associated with the pool.
  std::shared_ptr<platform_impl> MPlatform;

  /// Counter for allocations done by the pool that are currently alive. This
  /// includes managed resources that are currently alive.
  size_t MAllocCount = 0;

  /// A set of all free entries in the pool.
  std::multiset<FreeEntry, FreeEntryCompare> MFreeEntries;

  /// Mutex protecting access to the pool.
  std::mutex MMutex;
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
