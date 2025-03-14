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

ur_usm_pool_handle_t
create_memory_pool_device(const sycl::context &ctx, const sycl::device &dev,
                          const size_t threshold, const size_t maxSize,
                          const bool readOnly, const bool zeroInit) {

  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(ctx);
  ur_context_handle_t C = CtxImpl->getHandleRef();
  std::shared_ptr<sycl::detail::device_impl> DevImpl =
      sycl::detail::getSyclObjImpl(dev);
  ur_device_handle_t Device = DevImpl->getHandleRef();
  const sycl::detail::AdapterPtr &Adapter = CtxImpl->getAdapter();

  ur_usm_pool_limits_desc_t LimitsDesc{UR_STRUCTURE_TYPE_USM_POOL_LIMITS_DESC,
                                       nullptr, maxSize, threshold};

  ur_usm_pool_flags_t Flags = {UR_USM_POOL_FLAG_USE_NATIVE_MEMORY_POOL_EXP};
  if (readOnly)
    Flags += UR_USM_POOL_FLAG_READ_ONLY_EXP;
  if (zeroInit)
    Flags += UR_USM_POOL_FLAG_ZERO_INITIALIZE_BLOCK;

  ur_usm_pool_desc_t PoolDesc{UR_STRUCTURE_TYPE_USM_POOL_DESC, &LimitsDesc,
                              Flags};

  ur_usm_pool_handle_t poolHandle;

  Adapter
      ->call<sycl::errc::runtime, sycl::detail::UrApiKind::urUSMPoolCreateExp>(
          C, Device, &PoolDesc, &poolHandle);

  return poolHandle;
}

void destroy_memory_pool(const sycl::context &ctx, const sycl::device &dev,
                         ur_usm_pool_handle_t &poolHandle) {

  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(ctx);
  ur_context_handle_t C = CtxImpl->getHandleRef();
  std::shared_ptr<sycl::detail::device_impl> DevImpl =
      sycl::detail::getSyclObjImpl(dev);
  ur_device_handle_t Device = DevImpl->getHandleRef();
  const sycl::detail::AdapterPtr &Adapter = CtxImpl->getAdapter();

  Adapter
      ->call<sycl::errc::runtime, sycl::detail::UrApiKind::urUSMPoolDestroyExp>(
          C, Device, poolHandle);
}
} // namespace

// <--- Memory pool impl --->
detail::memory_pool_impl::memory_pool_impl(const sycl::context &ctx,
                                           const sycl::device &dev,
                                           const sycl::usm::alloc kind,
                                           const property_list &props)
    : MContextImplPtr(sycl::detail::getSyclObjImpl(ctx)), MDevice(dev),
      MKind(kind), MPropList(props) {
  size_t maxSize = 0;
  size_t threshold = 0;
  bool readOnly = false;
  bool zeroInit = false;

  // Get properties.
  if (props.has_property<property::maximum_size>()) {
    maxSize = props.get_property<property::maximum_size>().get_maximum_size();
  }
  if (props.has_property<property::initial_threshold>()) {
    threshold = props.get_property<property::initial_threshold>()
                    .get_initial_threshold();
  }
  if (props.has_property<property::read_only>()) {
    readOnly = true;
  }
  if (props.has_property<property::zero_init>()) {
    zeroInit = true;
  }

  if (kind == sycl::usm::alloc::device)
    MPoolHandle = create_memory_pool_device(ctx, dev, threshold, maxSize,
                                            readOnly, zeroInit);
  else
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Only device allocated memory pools are supported!");
}

detail::memory_pool_impl::memory_pool_impl(const sycl::context &ctx,
                                           const sycl::device &dev,
                                           const sycl::usm::alloc kind,
                                           ur_usm_pool_handle_t poolHandle,
                                           const bool isDefaultPool,
                                           const property_list &props)
    : MContextImplPtr(sycl::detail::getSyclObjImpl(ctx)), MDevice(dev),
      MKind(kind), MPoolHandle(poolHandle), MIsDefaultPool(isDefaultPool),
      MPropList(props) {}

detail::memory_pool_impl::~memory_pool_impl() {

  // Default memory pools cannot be destroyed.
  if (MIsDefaultPool) {
    return;
  }
  ur_usm_pool_handle_t handle = this->get_handle();
  sycl::context ctx = this->get_context();
  sycl::device dev = this->get_device();
  destroy_memory_pool(ctx, dev, handle);
}

size_t detail::memory_pool_impl::get_threshold() const {
  const sycl::detail::AdapterPtr &Adapter = MContextImplPtr->getAdapter();

  size_t threshold = 0;
  Adapter
      ->call<sycl::errc::runtime, sycl::detail::UrApiKind::urUSMPoolGetInfoExp>(
          MPoolHandle, UR_USM_POOL_INFO_RELEASE_THRESHOLD_EXP, &threshold,
          nullptr);

  return threshold;
}

size_t detail::memory_pool_impl::get_reserved_size_current() const {
  const sycl::detail::AdapterPtr &Adapter = MContextImplPtr->getAdapter();

  size_t resSizeCurrent = 0;
  Adapter
      ->call<sycl::errc::runtime, sycl::detail::UrApiKind::urUSMPoolGetInfoExp>(
          MPoolHandle, UR_USM_POOL_INFO_RESERVED_CURRENT_EXP, &resSizeCurrent,
          nullptr);

  return resSizeCurrent;
}

size_t detail::memory_pool_impl::get_reserved_size_high() const {
  const sycl::detail::AdapterPtr &Adapter = MContextImplPtr->getAdapter();

  size_t resSizeHigh = 0;
  Adapter
      ->call<sycl::errc::runtime, sycl::detail::UrApiKind::urUSMPoolGetInfoExp>(
          MPoolHandle, UR_USM_POOL_INFO_RESERVED_HIGH_EXP, &resSizeHigh,
          nullptr);

  return resSizeHigh;
}

size_t detail::memory_pool_impl::get_used_size_current() const {
  const sycl::detail::AdapterPtr &Adapter = MContextImplPtr->getAdapter();

  size_t usedSizeCurrent = 0;
  Adapter
      ->call<sycl::errc::runtime, sycl::detail::UrApiKind::urUSMPoolGetInfoExp>(
          MPoolHandle, UR_USM_POOL_INFO_USED_CURRENT_EXP, &usedSizeCurrent,
          nullptr);

  return usedSizeCurrent;
}

size_t detail::memory_pool_impl::get_used_size_high() const {
  const sycl::detail::AdapterPtr &Adapter = MContextImplPtr->getAdapter();

  size_t usedSizeHigh = 0;
  Adapter
      ->call<sycl::errc::runtime, sycl::detail::UrApiKind::urUSMPoolGetInfoExp>(
          MPoolHandle, UR_USM_POOL_INFO_USED_HIGH_EXP, &usedSizeHigh, nullptr);

  return usedSizeHigh;
}

void detail::memory_pool_impl::set_new_threshold(size_t newThreshold) {
  const sycl::detail::AdapterPtr &Adapter = MContextImplPtr->getAdapter();

  Adapter
      ->call<sycl::errc::runtime, sycl::detail::UrApiKind::urUSMPoolSetInfoExp>(
          MPoolHandle, UR_USM_POOL_INFO_RELEASE_THRESHOLD_EXP, &newThreshold,
          8 /*uint64_t*/);
}

void detail::memory_pool_impl::reset_reserved_size_high() {
  const sycl::detail::AdapterPtr &Adapter = MContextImplPtr->getAdapter();

  uint64_t resetVal = 0; // Reset to zero
  Adapter
      ->call<sycl::errc::runtime, sycl::detail::UrApiKind::urUSMPoolSetInfoExp>(
          MPoolHandle, UR_USM_POOL_INFO_RESERVED_HIGH_EXP,
          static_cast<void *>(&resetVal), 8 /*uint64_t*/);
}

void detail::memory_pool_impl::reset_used_size_high() {
  const sycl::detail::AdapterPtr &Adapter = MContextImplPtr->getAdapter();

  uint64_t resetVal = 0; // Reset to zero
  Adapter
      ->call<sycl::errc::runtime, sycl::detail::UrApiKind::urUSMPoolSetInfoExp>(
          MPoolHandle, UR_USM_POOL_INFO_USED_HIGH_EXP,
          static_cast<void *>(&resetVal), 8 /*uint64_t*/);
}

void detail::memory_pool_impl::trim_to(size_t minBytesToKeep) {
  ur_context_handle_t C = MContextImplPtr->getHandleRef();
  std::shared_ptr<sycl::detail::device_impl> DevImpl =
      sycl::detail::getSyclObjImpl(MDevice);
  ur_device_handle_t Device = DevImpl->getHandleRef();
  const sycl::detail::AdapterPtr &Adapter = MContextImplPtr->getAdapter();

  Adapter
      ->call<sycl::errc::runtime, sycl::detail::UrApiKind::urUSMPoolTrimToExp>(
          C, Device, MPoolHandle, minBytesToKeep);
}

// <--- Memory pool --->
__SYCL_EXPORT size_t memory_pool::get_threshold() const {
  return impl->get_threshold();
}

__SYCL_EXPORT const property_list &memory_pool::getPropList() const {
  return impl->getPropList();
}

__SYCL_EXPORT size_t memory_pool::get_reserved_size_current() const {
  return impl->get_reserved_size_current();
}

__SYCL_EXPORT size_t memory_pool::get_reserved_size_high() const {
  return impl->get_reserved_size_high();
}

__SYCL_EXPORT size_t memory_pool::get_used_size_current() const {
  return impl->get_used_size_current();
}

__SYCL_EXPORT size_t memory_pool::get_used_size_high() const {
  return impl->get_used_size_high();
}

__SYCL_EXPORT void memory_pool::set_new_threshold(size_t newThreshold) {

  // Throw when threshold is being reduced
  if (newThreshold < get_threshold()) {
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "Lowering the release threshold is disallowed!");
  }
  impl->set_new_threshold(newThreshold);
}

__SYCL_EXPORT void memory_pool::reset_reserved_size_high() {
  impl->reset_reserved_size_high();
}

__SYCL_EXPORT void memory_pool::reset_used_size_high() {
  impl->reset_used_size_high();
}

__SYCL_EXPORT void memory_pool::trim_to(size_t minBytesToKeep) {
  return impl->trim_to(minBytesToKeep);
}

__SYCL_EXPORT memory_pool::memory_pool(const sycl::context &ctx,
                                       const sycl::device &dev,
                                       const sycl::usm::alloc kind,
                                       const property_list &props) {

  if (kind == sycl::usm::alloc::host) {
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::invalid),
        "Host allocated memory pools selected but device supplied!");
  }

  if (kind != sycl::usm::alloc::device) {
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Only device allocated memory pools are supported!");
  }

  impl = std::make_shared<detail::memory_pool_impl>(ctx, dev, kind, props);
}

// NOT SUPPORTED: Host side pools unsupported.
__SYCL_EXPORT memory_pool::memory_pool(const sycl::context &,
                                       const property_list &) {
  throw sycl::exception(
      sycl::make_error_code(sycl::errc::feature_not_supported),
      "Host allocated pools are unsupported!");
}

__SYCL_EXPORT
memory_pool::memory_pool(const sycl::queue &q, const sycl::usm::alloc kind,
                         const property_list &props)
    : memory_pool(q.get_context(), q.get_device(), kind, props) {}

// NOT SUPPORTED: Creating a pool from an existing allocation is unsupported.
__SYCL_EXPORT memory_pool::memory_pool(const sycl::context &, const void *,
                                       size_t, const property_list &) {
  throw sycl::exception(
      sycl::make_error_code(sycl::errc::feature_not_supported),
      "Creating a pool from an existing allocation is unsupported!");
}

// <--- Async allocs/frees --->
__SYCL_EXPORT void *async_malloc(sycl::handler &h, sycl::usm::alloc kind,
                                 size_t size) {

  // Non-device allocations are unsupported.
  if (kind != sycl::usm::alloc::device) {
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Only device backed asynchronous allocations are supported!");
  }

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
  // Set up CommandGroup which is a no-op and pass the event from the alloc.
  h.impl->MAllocSize = size;
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
                                           memory_pool &pool) {

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
  h.impl->MAllocSize = size;
  h.impl->MMemPool = memPoolImpl;
  h.impl->MAsyncAllocEvent = Event;
  h.setType(detail::CGType::AsyncAlloc);

  return alloc;
}

__SYCL_EXPORT void *
async_malloc_from_pool(const sycl::queue &q, size_t size, memory_pool &pool,
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
