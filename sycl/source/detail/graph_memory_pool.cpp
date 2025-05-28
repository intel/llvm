//==--------- graph_memory_pool.cpp --- SYCL graph extension ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "graph_memory_pool.hpp"

#include <optional>
#include <queue>

#include "graph_impl.hpp"

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {
namespace detail {

void *
graph_mem_pool::malloc(size_t Size, usm::alloc AllocType,
                       const std::vector<std::shared_ptr<node_impl>> &DepNodes,
                       const std::shared_ptr<memory_pool_impl> &MemPool) {
  // We are potentially modifying contents of this memory pool and the owning
  // graph, so take a lock here.
  graph_impl::WriteLock Lock(MGraph.MMutex);

  void *Alloc = nullptr;
  alloc_info AllocInfo = {};
  AllocInfo.Kind = AllocType;
  // Collect relevant properties from memory pool
  if (MemPool) {
    auto Props = MemPool->getProps();
    if (Props.zero_init.second) {
      AllocInfo.ZeroInit = true;
    }
    if (Props.read_only.second) {
      AllocInfo.ReadOnly = true;
    }
  }

  switch (AllocType) {
  case usm::alloc::device: {

    auto &CtxImpl = sycl::detail::getSyclObjImpl(MContext);
    auto &Adapter = CtxImpl->getAdapter();

    size_t Granularity = get_mem_granularity(MDevice, MContext);
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
    Adapter->call<sycl::errc::runtime,
                  sycl::detail::UrApiKind::urVirtualMemReserve>(
        CtxImpl->getHandleRef(), reinterpret_cast<void *>(StartPtr),
        AlignedSize, &Alloc);

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
graph_mem_pool::tryReuseExistingAllocation(
    size_t Size, usm::alloc AllocType, bool ReadOnly,
    const std::vector<std::shared_ptr<node_impl>> &DepNodes) {
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

  std::queue<std::weak_ptr<node_impl>> NodesToCheck;

  // Add all the dependent nodes to the queue, they will be popped first
  for (auto &Dep : DepNodes) {
    NodesToCheck.push(Dep);
  }

  // Called when traversing over nodes to check if the current node is a free
  // node for one of the available allocations. If it is we populate AllocInfo
  // with the allocation to be reused.
  auto CheckNodeEqual =
      [&CompatibleAllocs](const std::shared_ptr<node_impl> &CurrentNode)
      -> std::optional<alloc_info> {
    for (auto &Alloc : CompatibleAllocs) {
      const auto &AllocFreeNode = Alloc.LastFreeNode;
      // Compare control blocks without having to lock AllocFreeNode to check
      // for node equality
      if (!CurrentNode.owner_before(AllocFreeNode) &&
          !AllocFreeNode.owner_before(CurrentNode)) {
        return Alloc;
      }
    }
    return std::nullopt;
  };

  while (!NodesToCheck.empty()) {
    auto CurrentNode = NodesToCheck.front().lock();
    NodesToCheck.pop();

    if (CurrentNode->MTotalVisitedEdges > 0) {
      continue;
    }

    // Check if the node is a free node and, if so, check if it is a free node
    // for any of the allocations which are free for reuse. We should not bother
    // checking nodes that are not free nodes, so we continue and check their
    // predecessors.
    if (CurrentNode->MNodeType == node_type::async_free) {
      std::optional<alloc_info> AllocFound = CheckNodeEqual(CurrentNode);
      if (AllocFound) {
        // Reset visited nodes tracking
        MGraph.resetNodeVisitedEdges();
        // Reset last free node for allocation
        MAllocations.at(AllocFound.value().Ptr).LastFreeNode.reset();
        // Remove found allocation from the free list
        MFreeAllocations.erase(std::find(MFreeAllocations.begin(),
                                         MFreeAllocations.end(),
                                         AllocFound.value().Ptr));
        return AllocFound;
      }
    }

    // Add CurrentNode predecessors to queue
    for (auto &Pred : CurrentNode->MPredecessors) {
      NodesToCheck.push(Pred);
    }

    // Mark node as visited
    CurrentNode->MTotalVisitedEdges = 1;
  }

  return std::nullopt;
}

void graph_mem_pool::markAllocationAsAvailable(
    void *Ptr, const std::shared_ptr<node_impl> &FreeNode) {
  MFreeAllocations.push_back(Ptr);
  MAllocations.at(Ptr).LastFreeNode = FreeNode;
}

} // namespace detail
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
