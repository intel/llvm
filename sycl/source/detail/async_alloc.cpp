//==----------- async_alloc.cpp --- SYCL asynchronous allocation -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/ur.hpp>
#include <sycl/ext/oneapi/async_alloc/async_alloc.hpp>

#include <detail/context_impl.hpp>
#include <detail/queue_impl.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

ur_usm_pool_handle_t create_memory_pool_device(const sycl::context &ctx,
                                               const sycl::device &dev,
                                               const size_t threshold,
                                               const size_t maxSize) {

  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(ctx);
  ur_context_handle_t C = CtxImpl->getHandleRef();
  std::shared_ptr<sycl::detail::device_impl> DevImpl =
      sycl::detail::getSyclObjImpl(dev);
  ur_device_handle_t Device = DevImpl->getHandleRef();
  const sycl::detail::AdapterPtr &Adapter = CtxImpl->getAdapter();

  ur_usm_pool_limits_desc_t LimitsDesc{UR_STRUCTURE_TYPE_USM_POOL_LIMITS_DESC,
                                       nullptr, maxSize, threshold};

  ur_usm_pool_desc_t PoolDesc{UR_STRUCTURE_TYPE_USM_POOL_DESC, &LimitsDesc,
                              UR_USM_POOL_FLAG_USE_NATIVE_MEMORY_POOL_EXP};

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

// <--- Memory pool impl --->
detail::memory_pool_impl::memory_pool_impl(const sycl::context &ctx,
                                           const sycl::device &dev,
                                           const sycl::usm::alloc kind,
                                           const size_t threshold,
                                           const size_t maxSize,
                                           const property_list &props)
    : syclContext(ctx), syclDevice(dev), kind(kind), threshold(threshold),
      maxSize(maxSize), propList(props) {

  if (kind == sycl::usm::alloc::device)
    poolHandle = create_memory_pool_device(ctx, dev, threshold, maxSize);
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
    : syclContext(ctx), syclDevice(dev), kind(kind), poolHandle(poolHandle),
      isDefaultPool(true), propList(props) {}

detail::memory_pool_impl::~memory_pool_impl() {

  // Default memory pools cannot be destroyed
  if (isDefaultPool)
    return;
  ur_usm_pool_handle_t handle = this->get_handle();
  sycl::context ctx = this->get_context();
  sycl::device dev = this->get_device();
  destroy_memory_pool(ctx, dev, handle);
}

void detail::memory_pool_impl::set_new_threshold(size_t newThreshold) {

  if (newThreshold > get_max_size()) {
    newThreshold = get_max_size();
  }

  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  ur_context_handle_t C = CtxImpl->getHandleRef();
  std::shared_ptr<sycl::detail::device_impl> DevImpl =
      sycl::detail::getSyclObjImpl(syclDevice);
  ur_device_handle_t Device = DevImpl->getHandleRef();
  const sycl::detail::AdapterPtr &Adapter = CtxImpl->getAdapter();

  Adapter->call<sycl::errc::runtime,
                sycl::detail::UrApiKind::urUSMPoolSetThresholdExp>(
      C, Device, poolHandle, newThreshold);

  threshold = newThreshold;
}

// <--- Memory pool --->
__SYCL_EXPORT void memory_pool::set_new_threshold(size_t newThreshold) {
  impl->set_new_threshold(newThreshold);
}

__SYCL_EXPORT const property_list &memory_pool::getPropList() const {
  return impl->getPropList();
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

  size_t maxSize = 0;
  size_t initialThreshold = 0;

  // Get props
  if (props.has_property<property::maximum_size>()) {
    maxSize = props.get_property<property::maximum_size>().get_maximum_size();
  }

  if (props.has_property<property::initial_threshold>()) {

    initialThreshold = props.get_property<property::initial_threshold>()
                           .get_initial_threshold();
  }

  std::cout << "initial threshold: " << initialThreshold << std::endl;
  std::cout << "maximum size: " << maxSize << std::endl;

  // Currently not sure how to handle this
  if (props.has_property<property::read_only>()) {
    std::cout << "Read Only property selected!" << std::endl;
  }

  // Currently CUDA does this automatically for us
  if (props.has_property<property::zero_init>()) {
    std::cout << "Zero init property selected!" << std::endl;
  }

  // Impl
  impl = std::make_shared<detail::memory_pool_impl>(
      ctx, dev, kind, initialThreshold, maxSize, props);
}

// NOT SUPPORTED: Host side pools unsupported
__SYCL_EXPORT memory_pool::memory_pool(const sycl::context &ctx,
                                       const property_list &props) {

  throw sycl::exception(
      sycl::make_error_code(sycl::errc::feature_not_supported),
      "Host allocated pools are unsupported!");
}

__SYCL_EXPORT
memory_pool::memory_pool(const sycl::queue &q, const sycl::usm::alloc kind,
                         const property_list &props)
    : memory_pool(q.get_context(), q.get_device(), kind, props) {}

// NOT SUPPORTED: Creating a pool from an existing allocation is unsupported
__SYCL_EXPORT memory_pool::memory_pool(const sycl::context &ctx,
                                       const void *ptr, size_t size,
                                       const property_list &props) {
  throw sycl::exception(
      sycl::make_error_code(sycl::errc::feature_not_supported),
      "Creating a pool from an existing allocation is unsupported!");
}

// <--- Async allocs/frees --->
__SYCL_EXPORT
void *async_malloc_from_pool(const sycl::queue &q, size_t size,
                             memory_pool &pool) {

  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(q.get_context());
  const sycl::detail::AdapterPtr &Adapter = CtxImpl->getAdapter();
  std::shared_ptr<sycl::detail::queue_impl> QImpl =
      sycl::detail::getSyclObjImpl(q);
  ur_queue_handle_t Q = QImpl->getHandleRef();
  std::shared_ptr<detail::memory_pool_impl> memPoolImpl =
      sycl::detail::getSyclObjImpl(pool);

  void *alloc = nullptr;

  Adapter->call<sycl::errc::runtime,
                sycl::detail::UrApiKind::urEnqueueUSMDeviceAllocExp>(
      Q, memPoolImpl.get()->get_handle(), size, nullptr, 0, nullptr, &alloc,
      nullptr);

  return alloc;
}

__SYCL_EXPORT void async_free(const sycl::queue &q, void *ptr) {

  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(q.get_context());
  const sycl::detail::AdapterPtr &Adapter = CtxImpl->getAdapter();
  std::shared_ptr<sycl::detail::queue_impl> QImpl =
      sycl::detail::getSyclObjImpl(q);
  ur_queue_handle_t Q = QImpl->getHandleRef();

  Adapter
      ->call<sycl::errc::runtime, sycl::detail::UrApiKind::urEnqueueUSMFreeExp>(
          Q, nullptr, ptr, 0, nullptr, nullptr);
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
