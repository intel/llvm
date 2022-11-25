//==------------ fusion_wrapper.cpp ----------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/fusion/fusion_wrapper_impl.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

fusion_wrapper_impl::fusion_wrapper_impl(
    std::shared_ptr<detail::queue_impl> Queue)
    : MQueue{std::move(Queue)} {}

std::shared_ptr<detail::queue_impl> fusion_wrapper_impl::get_queue() const {
  return MQueue;
}

bool fusion_wrapper_impl::is_in_fusion_mode() const { return false; }

void fusion_wrapper_impl::start_fusion() {
  throw sycl::exception(sycl::errc::feature_not_supported,
                        "Fusion not yet implemented");
}

void fusion_wrapper_impl::cancel_fusion() {
  throw sycl::exception(sycl::errc::feature_not_supported,
                        "Fusion not yet implemented");
}

event fusion_wrapper_impl::complete_fusion(const property_list &PropList) {
  (void)PropList;
  throw sycl::exception(sycl::errc::feature_not_supported,
                        "Fusion not yet implemented");
}

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
