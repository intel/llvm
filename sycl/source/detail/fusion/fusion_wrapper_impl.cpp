//==------------ fusion_wrapper.cpp ----------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/fusion/fusion_wrapper_impl.hpp>

#include <detail/scheduler/scheduler.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

fusion_wrapper_impl::fusion_wrapper_impl(
    std::shared_ptr<detail::queue_impl> Queue)
    : MQueue{std::move(Queue)} {}

std::shared_ptr<detail::queue_impl> fusion_wrapper_impl::get_queue() const {
  return MQueue;
}

bool fusion_wrapper_impl::is_in_fusion_mode() const {
  return MQueue->is_in_fusion_mode();
}

void fusion_wrapper_impl::start_fusion() {
  detail::Scheduler::getInstance().startFusion(MQueue);
}

void fusion_wrapper_impl::cancel_fusion() {
  detail::Scheduler::getInstance().cancelFusion(MQueue);
}

event fusion_wrapper_impl::complete_fusion(const property_list &PropList) {
  auto EventImpl =
      detail::Scheduler::getInstance().completeFusion(MQueue, PropList);
  return detail::createSyclObjFromImpl<event>(EventImpl);
}

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
