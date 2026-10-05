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
#include <detail/graph/graph_impl.hpp>
#include <detail/graph/node_impl.hpp>
#include <detail/queue_impl.hpp>
#include <sycl/detail/ur.hpp>
#include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#include <sycl/ext/oneapi/experimental/async_alloc/memory_pool.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

namespace {
std::vector<ur_event_handle_t> getUrEvents(detail::events_range DepEvents) {
  std::vector<ur_event_handle_t> RetUrEvents;
  for (detail::event_impl &Event : DepEvents) {
    ur_event_handle_t Handle = Event.getHandle();
    if (Handle != nullptr)
      RetUrEvents.push_back(Handle);
  }
  return RetUrEvents;
}

std::vector<detail::node_impl *> getDepGraphNodes(
    sycl::handler &Handler, detail::queue_impl *Queue,
    const std::shared_ptr<detail::graph_impl> &Graph,
    const std::vector<std::shared_ptr<detail::event_impl>> &DepEvents) {
  detail::handler_impl &HandlerImpl = *detail::getSyclObjImpl(Handler);
  // Get dependent graph nodes from any events
  auto DepNodes = Graph->getNodesForEvents(DepEvents);
  // If this node was added explicitly we may have node deps in the handler as
  // well, so add them to the list
  for (auto &N : HandlerImpl.MNodeDeps)
    DepNodes.push_back(N.get());
  // If this is being recorded from an in-order queue we need to get the last
  // in-order node if any, since this will later become a dependency of the
  // node being processed here.
  if (detail::node_impl *LastInOrderNode = Graph->getLastInorderNode(Queue);
      LastInOrderNode) {
    DepNodes.push_back(LastInOrderNode);
  }
  return DepNodes;
}

void checkAsyncMallocKind(sycl::usm::alloc Kind) {
  if (Kind == sycl::usm::alloc::unknown)
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "Unknown allocation kinds are disallowed!");

  // Non-device allocations are unsupported.
  if (Kind != sycl::usm::alloc::device)
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Only device backed asynchronous allocations are supported!");
}

void checkNotNativeRecording(detail::queue_impl &Queue, const char *FuncName) {
  // Allocations are not supported in graph native recording mode.
  if (Queue.isNativeRecording())
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          std::string(FuncName) +
                              " is not supported in native recording mode.");
}
} // namespace

__SYCL_EXPORT
void *async_malloc(sycl::handler &h, sycl::usm::alloc kind, size_t size) {

  checkAsyncMallocKind(kind);

  if (auto *Queue = h.impl->get_queue_or_null(); Queue)
    checkNotNativeRecording(*Queue, "async_malloc");

  detail::adapter_impl &Adapter = h.getContextImpl().getAdapter();

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
    Adapter.call<sycl::errc::runtime,
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
  detail::queue_impl &Queue = *detail::getSyclObjImpl(q);

  // Allocations recorded to a SYCL command-graph are served from the graph's
  // own memory pool, which requires a graph node, and therefore a handler. The
  // check has to be made here rather than inside the submission, as a handler
  // cannot be created while the queue is locked.
  if (Queue.hasCommandGraph()) {
    void *temp = nullptr;
    submit(
        q, [&](sycl::handler &h) { temp = async_malloc(h, kind, size); },
        CodeLoc);
    return temp;
  }

  checkAsyncMallocKind(kind);
  checkNotNativeRecording(Queue, "async_malloc");

  return Queue.submit_async_malloc_direct(/*Pool*/ nullptr, size, CodeLoc);
}

__SYCL_EXPORT void *async_malloc_from_pool(sycl::handler &h, size_t size,
                                           const memory_pool &pool) {

  if (auto *Queue = h.impl->get_queue_or_null(); Queue)
    checkNotNativeRecording(*Queue, "async_malloc_from_pool");

  detail::adapter_impl &Adapter = h.getContextImpl().getAdapter();
  detail::memory_pool_impl &memPoolImpl = *detail::getSyclObjImpl(pool);

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
                                       detail::getSyclObjImpl(pool).get());
  } else {
    ur_queue_handle_t Q = h.impl->get_queue().getHandleRef();
    Adapter.call<sycl::errc::runtime,
                 sycl::detail::UrApiKind::urEnqueueUSMDeviceAllocExp>(
        Q, memPoolImpl.get_handle(), size, nullptr, UREvents.size(),
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
  detail::queue_impl &Queue = *detail::getSyclObjImpl(q);

  // Allocations recorded to a SYCL command-graph are served from the graph's
  // own memory pool, which requires a graph node, and therefore a handler.
  if (Queue.hasCommandGraph()) {
    void *temp = nullptr;
    submit(
        q,
        [&](sycl::handler &h) { temp = async_malloc_from_pool(h, size, pool); },
        CodeLoc);
    return temp;
  }

  checkNotNativeRecording(Queue, "async_malloc_from_pool");

  return Queue.submit_async_malloc_direct(
      detail::getSyclObjImpl(pool)->get_handle(), size, CodeLoc);
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

  if (auto *Queue = h.impl->get_queue_or_null(); Queue)
    checkNotNativeRecording(*Queue, "async_free");

  h.impl->MFreePtr = ptr;
  h.setType(detail::CGType::AsyncFree);
}

__SYCL_EXPORT void async_free(const sycl::queue &q, void *ptr,
                              const sycl::detail::code_location &CodeLoc) {
  detail::queue_impl &Queue = *detail::getSyclObjImpl(q);

  // Frees recorded to a SYCL command-graph operate on the graph's own memory
  // pool, which requires a graph node, and therefore a handler.
  if (Queue.hasCommandGraph()) {
    submit(q, [&](sycl::handler &h) { async_free(h, ptr); }, CodeLoc);
    return;
  }

  checkNotNativeRecording(Queue, "async_free");

  Queue.submit_async_free_direct(ptr, CodeLoc);
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
