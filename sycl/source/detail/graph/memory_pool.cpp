//==--------- memory_pool.cpp --- SYCL graph extension ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "memory_pool.hpp"
#include "detail/virtual_mem.hpp"
#include "graph_impl.hpp"

#include <optional>
#include <queue>

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {
namespace detail {

void *graph_mem_pool::malloc(size_t Size, usm::alloc AllocType,
                             nodes_range DepNodes, memory_pool_impl *MemPool) {
  // We are potentially modifying contents of this memory pool and the owning
  // graph, so take a lock here.
  graph_impl::WriteLock Lock(MGraph.MMutex);

  void *Alloc = nullptr;
  alloc_info AllocInfo = {};
  AllocInfo.Kind = AllocType;
  // Collect relevant properties from memory pool
  if (MemPool) {
    auto Props = MemPool->getProps();
    if (Props.zero_init) {
      AllocInfo.ZeroInit = true;
    }
  }

  switch (AllocType) {
  case usm::alloc::device: {

    const context_impl &CtxImpl = *getSyclObjImpl(MContext);
    const adapter_impl &Adapter = CtxImpl.getAdapter();
    const device_impl &DeviceImpl = *getSyclObjImpl(MDevice);

    const size_t Granularity = get_mem_granularity_for_allocation_size(
        DeviceImpl, CtxImpl, granularity_mode::recommended, Size);
    uintptr_t StartPtr = 0;
    size_t AlignedSize = alignByteSize(Size, Granularity);
    // See if we can find an allocation to reuse
    auto AllocOpt = tryReuseExistingAllocation(AlignedSize, AllocType,
                                               AllocInfo.ReadOnly, DepNodes);

    // If we got a value back then an allocation was available for reuse and we
    // can just return that pointer
    if (AllocOpt) {
      return AllocOpt.value().Ptr;
    }

    // If no allocation could be reused, do a new virtual reservation
    Adapter.call<sycl::errc::runtime,
                 sycl::detail::UrApiKind::urVirtualMemReserve>(
        CtxImpl.getHandleRef(), reinterpret_cast<void *>(StartPtr), AlignedSize,
        &Alloc);

    AllocInfo.Size = AlignedSize;
    AllocInfo.Ptr = Alloc;

    MAllocations[Alloc] = AllocInfo;
    break;
  }

  default:
    throw sycl::exception(sycl::make_error_code(errc::feature_not_supported),
                          "Only device allocations are currently supported "
                          "in graph allocation nodes!");
    break;
  }

  return Alloc;
}

std::optional<graph_mem_pool::alloc_info>
graph_mem_pool::tryReuseExistingAllocation(size_t Size, usm::alloc AllocType,
                                           bool ReadOnly,
                                           nodes_range DepNodes) {
  // If we have no dependencies this is a no-op because allocations must connect
  // to a free node for reuse to be possible.
  if (DepNodes.empty()) {
    return std::nullopt;
  }

  std::vector<alloc_info> CompatibleAllocs;
  // Compatible allocs can only be as big as MFreeAllocations
  CompatibleAllocs.reserve(MFreeAllocations.size());

  // Loop over free allocation list, search for ones that are compatible for
  // reuse. Currently that means they have the same alloc kind, size and read
  // only property.

  for (auto &Ptr : MFreeAllocations) {
    alloc_info &Info = MAllocations.at(Ptr);
    if (Info.Kind == AllocType && Info.Size == Size &&
        Info.ReadOnly == ReadOnly) {
      // Store the alloc info since it is compatible
      CompatibleAllocs.push_back(Info);
    }
  }

  // If we have no suitable allocs to reuse, return early
  if (CompatibleAllocs.size() == 0) {
    return std::nullopt;
  }

  // Traverse graph back from each DepNode to try and find any of the suitable
  // free nodes. We do this in a breadth-first approach because we want to find
  // the shortest path to a reusable allocation.

  // Add all the dependent nodes to the queue, they will be popped first
  auto NodesToCheck = DepNodes.to<std::queue<node_impl *>>();

  // Called when traversing over nodes to check if the current node is a free
  // node for one of the available allocations. If it is we populate AllocInfo
  // with the allocation to be reused.
  auto CheckNodeEqual =
      [&CompatibleAllocs](node_impl &CurrentNode) -> std::optional<alloc_info> {
    for (auto &Alloc : CompatibleAllocs) {
      if (&CurrentNode == Alloc.LastFreeNode) {
        return Alloc;
      }
    }
    return std::nullopt;
  };

  while (!NodesToCheck.empty()) {
    node_impl &CurrentNode = *NodesToCheck.front();

    if (CurrentNode.MTotalVisitedEdges > 0) {
      continue;
    }

    // Check if the node is a free node and, if so, check if it is a free node
    // for any of the allocations which are free for reuse. We should not bother
    // checking nodes that are not free nodes, so we continue and check their
    // predecessors.
    if (CurrentNode.MNodeType == node_type::async_free) {
      std::optional<alloc_info> AllocFound = CheckNodeEqual(CurrentNode);
      if (AllocFound) {
        // Reset visited nodes tracking
        MGraph.resetNodeVisitedEdges();
        // Reset last free node for allocation
        MAllocations.at(AllocFound.value().Ptr).LastFreeNode = nullptr;
        // Remove found allocation from the free list
        MFreeAllocations.erase(std::find(MFreeAllocations.begin(),
                                         MFreeAllocations.end(),
                                         AllocFound.value().Ptr));
        return AllocFound;
      }
    }

    // Add CurrentNode predecessors to queue
    for (node_impl &Pred : CurrentNode.predecessors()) {
      NodesToCheck.push(&Pred);
    }

    // Mark node as visited
    CurrentNode.MTotalVisitedEdges = 1;
    NodesToCheck.pop();
  }

  return std::nullopt;
}

void graph_mem_pool::markAllocationAsAvailable(void *Ptr, node_impl &FreeNode) {
  MFreeAllocations.push_back(Ptr);
  MAllocations.at(Ptr).LastFreeNode = &FreeNode;
}

} // namespace detail
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
