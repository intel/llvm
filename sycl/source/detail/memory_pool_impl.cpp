//==----------- memory_pool_impl.cpp --- SYCL asynchronous allocation ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/context_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/memory_pool_impl.hpp>
#include <detail/queue_impl.hpp>
#include <sycl/detail/ur.hpp>
#include <sycl/ext/oneapi/experimental/async_alloc/memory_pool.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
namespace detail {

// <--- Helpers --->
namespace {
ur_usm_pool_handle_t
create_memory_pool_device(const sycl::context &ctx, const sycl::device &dev,
                          const size_t threshold, const size_t maxSize,
                          const bool readOnly, const bool zeroInit) {
  auto [urDevice, urCtx, Adapter] = get_ur_handles(dev, ctx);

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
          urCtx, urDevice, &PoolDesc, &poolHandle);

  return poolHandle;
}

void destroy_memory_pool(const sycl::context &ctx, const sycl::device &dev,
                         ur_usm_pool_handle_t &poolHandle) {
  auto [urDevice, urCtx, Adapter] = get_ur_handles(dev, ctx);

  Adapter
      ->call<sycl::errc::runtime, sycl::detail::UrApiKind::urUSMPoolDestroyExp>(
          urCtx, urDevice, poolHandle);
}
} // namespace

// <--- Memory pool impl --->
memory_pool_impl::memory_pool_impl(const sycl::context &ctx,
                                   const sycl::device &dev,
                                   const sycl::usm::alloc kind,
                                   const memory_pool::pool_properties &props)
    : MContextImplPtr(sycl::detail::getSyclObjImpl(ctx)), MDevice(dev),
      MKind(kind), MProps(props) {

  if (kind == sycl::usm::alloc::device)
    MPoolHandle = create_memory_pool_device(
        ctx, dev, MProps.initial_threshold.second, MProps.maximum_size.second,
        MProps.read_only.second, MProps.zero_init.second);
  else
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Only device allocated memory pools are supported!");
}

memory_pool_impl::memory_pool_impl(const sycl::context &ctx,
                                   const sycl::device &dev,
                                   const sycl::usm::alloc kind,
                                   ur_usm_pool_handle_t poolHandle,
                                   const bool isDefaultPool,
                                   const memory_pool::pool_properties &props)
    : MContextImplPtr(sycl::detail::getSyclObjImpl(ctx)), MDevice(dev),
      MKind(kind), MPoolHandle(poolHandle), MIsDefaultPool(isDefaultPool),
      MProps(props) {}

memory_pool_impl::~memory_pool_impl() {

  // Default memory pools cannot be destroyed.
  if (MIsDefaultPool)
    return;

  try {
    ur_usm_pool_handle_t handle = this->get_handle();
    sycl::context ctx = this->get_context();
    sycl::device dev = this->get_device();
    destroy_memory_pool(ctx, dev, handle);
  } catch (std::exception &e) {
    __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~memory_pool_impl", e);
  }
}

size_t memory_pool_impl::get_threshold() const {
  const sycl::detail::AdapterPtr &Adapter = MContextImplPtr->getAdapter();

  size_t threshold = 0;
  Adapter
      ->call<sycl::errc::runtime, sycl::detail::UrApiKind::urUSMPoolGetInfoExp>(
          MPoolHandle, UR_USM_POOL_INFO_RELEASE_THRESHOLD_EXP, &threshold,
          nullptr);

  return threshold;
}

size_t memory_pool_impl::get_reserved_size_current() const {
  const sycl::detail::AdapterPtr &Adapter = MContextImplPtr->getAdapter();

  size_t resSizeCurrent = 0;
  Adapter
      ->call<sycl::errc::runtime, sycl::detail::UrApiKind::urUSMPoolGetInfoExp>(
          MPoolHandle, UR_USM_POOL_INFO_RESERVED_CURRENT_EXP, &resSizeCurrent,
          nullptr);

  return resSizeCurrent;
}

size_t memory_pool_impl::get_reserved_size_high() const {
  const sycl::detail::AdapterPtr &Adapter = MContextImplPtr->getAdapter();

  size_t resSizeHigh = 0;
  Adapter
      ->call<sycl::errc::runtime, sycl::detail::UrApiKind::urUSMPoolGetInfoExp>(
          MPoolHandle, UR_USM_POOL_INFO_RESERVED_HIGH_EXP, &resSizeHigh,
          nullptr);

  return resSizeHigh;
}

size_t memory_pool_impl::get_used_size_current() const {
  const sycl::detail::AdapterPtr &Adapter = MContextImplPtr->getAdapter();

  size_t usedSizeCurrent = 0;
  Adapter
      ->call<sycl::errc::runtime, sycl::detail::UrApiKind::urUSMPoolGetInfoExp>(
          MPoolHandle, UR_USM_POOL_INFO_USED_CURRENT_EXP, &usedSizeCurrent,
          nullptr);

  return usedSizeCurrent;
}

size_t memory_pool_impl::get_used_size_high() const {
  const sycl::detail::AdapterPtr &Adapter = MContextImplPtr->getAdapter();

  size_t usedSizeHigh = 0;
  Adapter
      ->call<sycl::errc::runtime, sycl::detail::UrApiKind::urUSMPoolGetInfoExp>(
          MPoolHandle, UR_USM_POOL_INFO_USED_HIGH_EXP, &usedSizeHigh, nullptr);

  return usedSizeHigh;
}

void memory_pool_impl::set_new_threshold(size_t newThreshold) {
  const sycl::detail::AdapterPtr &Adapter = MContextImplPtr->getAdapter();

  Adapter
      ->call<sycl::errc::runtime, sycl::detail::UrApiKind::urUSMPoolSetInfoExp>(
          MPoolHandle, UR_USM_POOL_INFO_RELEASE_THRESHOLD_EXP, &newThreshold,
          8 /*uint64_t*/);
}

void memory_pool_impl::reset_reserved_size_high() {
  const sycl::detail::AdapterPtr &Adapter = MContextImplPtr->getAdapter();

  uint64_t resetVal = 0; // Reset to zero
  Adapter
      ->call<sycl::errc::runtime, sycl::detail::UrApiKind::urUSMPoolSetInfoExp>(
          MPoolHandle, UR_USM_POOL_INFO_RESERVED_HIGH_EXP,
          static_cast<void *>(&resetVal), 8 /*uint64_t*/);
}

void memory_pool_impl::reset_used_size_high() {
  const sycl::detail::AdapterPtr &Adapter = MContextImplPtr->getAdapter();

  uint64_t resetVal = 0; // Reset to zero
  Adapter
      ->call<sycl::errc::runtime, sycl::detail::UrApiKind::urUSMPoolSetInfoExp>(
          MPoolHandle, UR_USM_POOL_INFO_USED_HIGH_EXP,
          static_cast<void *>(&resetVal), 8 /*uint64_t*/);
}

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
