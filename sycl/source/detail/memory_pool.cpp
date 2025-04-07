//==----------- memory_pool.cpp --- SYCL asynchronous allocation -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/memory_pool_impl.hpp>
#include <sycl/ext/oneapi/experimental/async_alloc/memory_pool.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

__SYCL_EXPORT sycl::context memory_pool::get_context() const {
  return impl->get_context();
}
__SYCL_EXPORT sycl::device memory_pool::get_device() const {
  return impl->get_device();
}
__SYCL_EXPORT sycl::usm::alloc memory_pool::get_alloc_kind() const {
  return impl->get_alloc_kind();
}

__SYCL_EXPORT size_t memory_pool::get_threshold() const {
  return impl->get_threshold();
}

const property_list &memory_pool::getPropList() const {
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

__SYCL_EXPORT void memory_pool::increase_threshold_to(size_t newThreshold) {
  // Only increase.
  if (newThreshold > get_threshold())
    impl->set_new_threshold(newThreshold);
}

__SYCL_EXPORT void memory_pool::reset_reserved_size_high() {
  impl->reset_reserved_size_high();
}

__SYCL_EXPORT void memory_pool::reset_used_size_high() {
  impl->reset_used_size_high();
}

__SYCL_EXPORT memory_pool::memory_pool(const sycl::context &ctx,
                                       const sycl::device &dev,
                                       sycl::usm::alloc kind,
                                       const property_list &props) {

  if (kind == sycl::usm::alloc::host)
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::invalid),
        "Host allocated memory pools selected but device supplied!");

  if (kind != sycl::usm::alloc::device)
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Only device allocated memory pools are supported!");

  impl = std::make_shared<detail::memory_pool_impl>(ctx, dev, kind, props);
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
