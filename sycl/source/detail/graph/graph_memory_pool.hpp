//==--------- graph_memory_pool.hpp --- SYCL graph extension ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "sycl/detail/common.hpp"
#include <detail/physical_mem_impl.hpp>                // For physical_mem_impl
#include <sycl/context.hpp>                            // For context
#include <sycl/device.hpp>                             // For device
#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp> // For get_mem_granularity

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {
namespace detail {

// Forward declarations
class node_impl;

/// Class handling graph-owned memory allocations. Device allocations are
/// managed using virtual memory.
class graph_mem_pool {
  // Info descriptor for various properties of an allocation
  struct alloc_info {
    // Pointer to the start of the allocation
    void *Ptr = nullptr;
    // Allocation kind
    usm::alloc Kind = usm::alloc::unknown;
    // Size of the allocation
    size_t Size = 0;
    // Is currently mapped to physical memory
    bool Mapped = false;
    // Index into the array of physical memory
    size_t PhysicalMemID = 0;
    // Is the allocation read only
    bool ReadOnly = false;
    // Should the allocation be zero initialized during initial allocation
    bool ZeroInit = false;
    // Last free node for this allocation in the graph
    std::weak_ptr<node_impl> LastFreeNode = {};
  };

public:
  graph_mem_pool(graph_impl &GraphImpl, const context &Context,
                 const device &Device)
      : MGraph(GraphImpl), MContext(Context), MDevice(Device) {}
  ~graph_mem_pool() {
    try {
      for (auto &[Ptr, AllocInfo] : MAllocations) {
        // Unmap allocations if required before physical memory is released
        // Physical mem is released when MPhysicalMem is cleared
        if (AllocInfo.Mapped) {
          unmap(Ptr, AllocInfo.Size, MContext);
        }
        // Free the VA range
        free_virtual_mem(reinterpret_cast<uintptr_t>(Ptr), AllocInfo.Size,
                         MContext);
      }
    } catch (std::exception &e) {
      __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~graph_mem_pool", e);
    }
    MPhysicalMem.clear();
  }

  /// Memory pool cannot be copied
  graph_mem_pool(graph_mem_pool &) = delete;

  /// Get a pointer to a new allocation. For device allocations these are
  /// virtual reservations which must be later mapped to allocated physical
  /// memory before use by calling allocateAndMapAll()
  /// @param Size Size of the allocation
  /// @param AllocType Type of the allocation
  /// @param DepNodes Any node dependencies for this allocation, used to
  /// identify allocations that can be reused.
  /// @param MemPool Optional memory pool from which allocations will not be
  /// made directly but properties may be respected.
  /// @return A pointer to the start of the allocation
  void *malloc(size_t Size, usm::alloc AllocType,
               const std::vector<std::shared_ptr<node_impl>> &DepNodes,
               const std::shared_ptr<memory_pool_impl> &MemPool = nullptr);

  /// Return the total amount of memory being used by this pool
  size_t getMemUseCurrent() const {
    size_t TotalMem = 0;
    for (auto &PhysicalMem : MPhysicalMem) {
      TotalMem += PhysicalMem->size();
    }

    return TotalMem;
  }

  /// For device allocations using virtual memory this function allocates
  /// physical memory and maps each virtual range to it, should be called during
  /// graph finalization.
  void allocateAndMapAll() {
    // Queue used for any initializing of memory, zero-init etc.
    sycl::queue Queue{MContext, MDevice};

    for (auto &Allocation : MAllocations) {
      // Set access mode
      void *Ptr = Allocation.first;
      alloc_info &AllocInfo = Allocation.second;
      address_access_mode AccessMode = AllocInfo.ReadOnly
                                           ? address_access_mode::read
                                           : address_access_mode::read_write;

      // Create physical memory
      auto PhysicalMem = std::make_shared<physical_mem_impl>(
          *getSyclObjImpl(MDevice), MContext, AllocInfo.Size);
      // Map the virtual reservation to it
      PhysicalMem->map(reinterpret_cast<uintptr_t>(Ptr), AllocInfo.Size,
                       AccessMode, 0);

      // Zero init if required
      if (AllocInfo.ZeroInit) {
        Queue.submit(
            [&](sycl::handler &CGH) { CGH.memset(Ptr, 0, AllocInfo.Size); });
      }

      MPhysicalMem.push_back(std::move(PhysicalMem));
      AllocInfo.PhysicalMemID = MPhysicalMem.size() - 1;
      AllocInfo.Mapped = true;
    }

    // Wait on any operations we enqueued.
    Queue.wait_and_throw();
  }

  /// For device virtual reservations unmap and deallocate physical memory.
  /// Virtual reservations are not released and can be reallocated/mapped again.
  /// Typically called on executable graph destruction.
  void deallocateAndUnmapAll() {
    for (auto &[Ptr, AllocInfo] : MAllocations) {
      // Unmap allocations before physical memory is released
      // Physical mem is released when MPhysicalMem is cleared
      unmap(Ptr, AllocInfo.Size, MContext);
      AllocInfo.PhysicalMemID = 0;
      AllocInfo.Mapped = false;
    }

    MPhysicalMem.clear();
  }

  /// True if this pool has any allocations
  bool hasAllocations() const { return MAllocations.size(); }

  // True if an allocation exists for this pointer
  bool hasAllocation(void *Ptr) const {
    return MAllocations.find(Ptr) != MAllocations.end();
  }

  /// Mark that this allocation has been freed and is available for reuse.
  /// @param Ptr The pointer to the allocation.
  /// @param FreeNode The graph node of node_type::async_free which is freeing
  /// the allocation.
  void markAllocationAsAvailable(void *Ptr,
                                 const std::shared_ptr<node_impl> &FreeNode);

private:
  /// Tries to reuse an existing allocation which has been marked free in the
  /// graph.
  /// @param Size Size of the allocation.
  /// @param AllocType USM type of the allocation.
  /// @param ReadOnly True if the allocation is read only
  /// @param DepNodes Node dependencies of this allocation, used to search for
  /// reusable allocations.
  /// @returns An optional allocation info value, where a null value indicates
  /// that no allocation could be reused.
  std::optional<alloc_info> tryReuseExistingAllocation(
      size_t Size, usm::alloc AllocType, bool ReadOnly,
      const std::vector<std::shared_ptr<node_impl>> &DepNodes);

  /// Returns an aligned byte size given a required granularity
  /// @param UnalignedByteSize The original requested allocation size
  /// @param Granularity The required granularity for this allocation
  /// @returns The aligned size
  static size_t alignByteSize(size_t UnalignedByteSize, size_t Granularity) {
    return ((UnalignedByteSize + Granularity - 1) / Granularity) * Granularity;
  }

  /// Graph that owns this mem pool
  graph_impl &MGraph;
  /// Context associated with allocations from this pool
  context MContext;
  /// Device associated with allocations from this pool
  device MDevice;
  /// Map of allocated pointers to an info struct
  std::unordered_map<void *, alloc_info> MAllocations;
  /// List of physical memory allocations used for virtual device reservations
  std::vector<std::shared_ptr<physical_mem_impl>> MPhysicalMem;
  /// List of pointers to allocations which are currently free for reuse
  std::vector<void *> MFreeAllocations;
};
} // namespace detail
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
