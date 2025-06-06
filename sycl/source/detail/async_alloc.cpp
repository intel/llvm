//==----------- async_alloc.cpp --- SYCL asynchronous allocation -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sycl/accessor.hpp"
#include <detail/context_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/graph_impl.hpp>
#include <detail/queue_impl.hpp>
#include <sycl/detail/ur.hpp>
#include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#include <sycl/ext/oneapi/experimental/async_alloc/memory_pool.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

namespace {
std::vector<ur_event_handle_t>
getUrEvents(const std::vector<std::shared_ptr<detail::event_impl>> &DepEvents) {
  std::vector<ur_event_handle_t> RetUrEvents;
  for (const std::shared_ptr<detail::event_impl> &EventImpl : DepEvents) {
    ur_event_handle_t Handle = EventImpl->getHandle();
    if (Handle != nullptr)
      RetUrEvents.push_back(Handle);
  }
  return RetUrEvents;
}

std::vector<std::shared_ptr<detail::node_impl>> getDepGraphNodes(
    sycl::handler &Handler, detail::queue_impl *Queue,
    const std::shared_ptr<detail::graph_impl> &Graph,
    const std::vector<std::shared_ptr<detail::event_impl>> &DepEvents) {
  auto HandlerImpl = detail::getSyclObjImpl(Handler);
  // Get dependent graph nodes from any events
  auto DepNodes = Graph->getNodesForEvents(DepEvents);
  // If this node was added explicitly we may have node deps in the handler as
  // well, so add them to the list
  DepNodes.insert(DepNodes.end(), HandlerImpl->MNodeDeps.begin(),
                  HandlerImpl->MNodeDeps.end());
  // If this is being recorded from an in-order queue we need to get the last
  // in-order node if any, since this will later become a dependency of the
  // node being processed here.
  if (const auto &LastInOrderNode = Graph->getLastInorderNode(Queue);
      LastInOrderNode) {
    DepNodes.push_back(LastInOrderNode);
  }
  return DepNodes;
}
} // namespace

__SYCL_EXPORT
void *async_malloc(sycl::handler &h, sycl::usm::alloc kind, size_t size) {

  if (kind == sycl::usm::alloc::unknown)
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "Unknown allocation kinds are disallowed!");

  // Non-device allocations are unsupported.
  if (kind != sycl::usm::alloc::device)
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Only device backed asynchronous allocations are supported!");

  auto &Adapter = h.getContextImplPtr()->getAdapter();

  // Get CG event dependencies for this allocation.
  const auto &DepEvents = h.impl->CGData.MEvents;
  auto UREvents = getUrEvents(DepEvents);

  void *alloc = nullptr;

  ur_event_handle_t Event = nullptr;
  // If a graph is present do the allocation from the graph memory pool instead.
  if (auto Graph = h.getCommandGraph(); Graph) {
    auto DepNodes =
        getDepGraphNodes(h, h.impl->get_queue_or_null(), Graph, DepEvents);
    alloc = Graph->getMemPool().malloc(size, kind, DepNodes);
  } else {
    ur_queue_handle_t Q = h.impl->get_queue().getHandleRef();
    Adapter->call<sycl::errc::runtime,
                  sycl::detail::UrApiKind::urEnqueueUSMDeviceAllocExp>(
        Q, (ur_usm_pool_handle_t)0, size, nullptr, UREvents.size(),
        UREvents.data(), &alloc, &Event);
  }

  // Async malloc must return a void* immediately.
  // Set up CommandGroup which is a no-op and pass the
  // event from the alloc.
  h.impl->MAsyncAllocEvent = Event;
  h.setType(detail::CGType::AsyncAlloc);

  return alloc;
}

__SYCL_EXPORT void *async_malloc(const sycl::queue &q, sycl::usm::alloc kind,
                                 size_t size,
                                 const sycl::detail::code_location &CodeLoc) {
  void *temp = nullptr;
  submit(
      q,
      [&](sycl::handler &h) {
        // In order queues must wait on the previous event before calling alloc.
        if (q.is_in_order() && q.ext_oneapi_get_last_event())
          h.depends_on(q.ext_oneapi_get_last_event().value());
        temp = async_malloc(h, kind, size);
      },
      CodeLoc);
  return temp;
}

__SYCL_EXPORT void *async_malloc_from_pool(sycl::handler &h, size_t size,
                                           const memory_pool &pool) {

  auto &Adapter = h.getContextImplPtr()->getAdapter();
  auto &memPoolImpl = sycl::detail::getSyclObjImpl(pool);

  // Get CG event dependencies for this allocation.
  const auto &DepEvents = h.impl->CGData.MEvents;
  auto UREvents = getUrEvents(DepEvents);

  void *alloc = nullptr;

  ur_event_handle_t Event = nullptr;
  // If a graph is present do the allocation from the graph memory pool instead.
  if (auto Graph = h.getCommandGraph(); Graph) {
    auto DepNodes =
        getDepGraphNodes(h, h.impl->get_queue_or_null(), Graph, DepEvents);

    // Memory pool is passed as the graph may use some properties of it.
    alloc = Graph->getMemPool().malloc(size, pool.get_alloc_kind(), DepNodes,
                                       sycl::detail::getSyclObjImpl(pool));
  } else {
    ur_queue_handle_t Q = h.impl->get_queue().getHandleRef();
    Adapter->call<sycl::errc::runtime,
                  sycl::detail::UrApiKind::urEnqueueUSMDeviceAllocExp>(
        Q, memPoolImpl.get()->get_handle(), size, nullptr, UREvents.size(),
        UREvents.data(), &alloc, &Event);
  }
  // Async malloc must return a void* immediately.
  // Set up CommandGroup which is a no-op and pass the event from the alloc.
  h.impl->MAsyncAllocEvent = Event;
  h.setType(detail::CGType::AsyncAlloc);

  return alloc;
}

__SYCL_EXPORT void *
async_malloc_from_pool(const sycl::queue &q, size_t size,
                       const memory_pool &pool,
                       const sycl::detail::code_location &CodeLoc) {
  void *temp = nullptr;
  submit(
      q,
      [&](sycl::handler &h) {
        // In order queues must wait on the previous event before calling alloc.
        if (q.is_in_order() && q.ext_oneapi_get_last_event())
          h.depends_on(q.ext_oneapi_get_last_event().value());
        temp = async_malloc_from_pool(h, size, pool);
      },
      CodeLoc);
  return temp;
}

__SYCL_EXPORT void async_free(sycl::handler &h, void *ptr) {
  // We only check for errors for the graph here because marking the allocation
  // as free in the graph memory pool requires a node object which doesn't exist
  // at this point.
  if (auto Graph = h.getCommandGraph(); Graph) {
    // Check if the pointer to be freed has an associated allocation node, and
    // error if not
    if (!Graph->getMemPool().hasAllocation(ptr)) {
      throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                            "Cannot add a free node to a graph for which "
                            "there is no associated allocation node!");
    }
  }

  h.impl->MFreePtr = ptr;
  h.setType(detail::CGType::AsyncFree);
}

__SYCL_EXPORT void async_free(const sycl::queue &q, void *ptr,
                              const sycl::detail::code_location &CodeLoc) {
  submit(q, [&](sycl::handler &h) { async_free(h, ptr); }, CodeLoc);
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
