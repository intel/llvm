//==----------- async_alloc.cpp --- SYCL asynchronous allocation -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/context_impl.hpp>
#include <detail/event_impl.hpp>
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

  h.throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_async_alloc>();

  auto &Adapter = h.getContextImplPtr()->getAdapter();
  auto &Q = h.MQueue->getHandleRef();

  // Get events to wait on.
  auto depEvents = getUrEvents(h.impl->CGData.MEvents);
  uint32_t numEvents = h.impl->CGData.MEvents.size();

  void *alloc = nullptr;
  ur_event_handle_t Event;
  Adapter->call<sycl::errc::runtime,
                sycl::detail::UrApiKind::urEnqueueUSMDeviceAllocExp>(
      Q, (ur_usm_pool_handle_t)0, size, nullptr, numEvents, depEvents.data(),
      &alloc, &Event);

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

  h.throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_async_alloc>();

  auto &Adapter = h.getContextImplPtr()->getAdapter();
  auto &Q = h.MQueue->getHandleRef();
  auto &memPoolImpl = sycl::detail::getSyclObjImpl(pool);

  // Get events to wait on.
  auto depEvents = getUrEvents(h.impl->CGData.MEvents);
  uint32_t numEvents = h.impl->CGData.MEvents.size();

  void *alloc = nullptr;
  ur_event_handle_t Event;
  Adapter->call<sycl::errc::runtime,
                sycl::detail::UrApiKind::urEnqueueUSMDeviceAllocExp>(
      Q, memPoolImpl.get()->get_handle(), size, nullptr, numEvents,
      depEvents.data(), &alloc, &Event);

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
  h.throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_async_alloc>();

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
