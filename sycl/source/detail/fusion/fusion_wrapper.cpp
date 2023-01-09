//==------------ fusion_wrapper.cpp ----------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/ext/codeplay/experimental/fusion_wrapper.hpp>

#include <detail/fusion/fusion_wrapper_impl.hpp>
#include <detail/queue_impl.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::codeplay::experimental {

fusion_wrapper::fusion_wrapper(queue &Queue) {
  if (!Queue.ext_codeplay_supports_fusion()) {
    throw sycl::exception(
        sycl::errc::invalid,
        "Cannot wrap a queue for fusion which doesn't support fusion");
  }
  MImpl = std::make_shared<detail::fusion_wrapper_impl>(
      sycl::detail::getSyclObjImpl(Queue));
}

queue fusion_wrapper::get_queue() const {
  return sycl::detail::createSyclObjFromImpl<sycl::queue>(MImpl->get_queue());
}

bool fusion_wrapper::is_in_fusion_mode() const {
  return MImpl->is_in_fusion_mode();
}

void fusion_wrapper::start_fusion() { MImpl->start_fusion(); }

void fusion_wrapper::cancel_fusion() { MImpl->cancel_fusion(); }

event fusion_wrapper::complete_fusion(const property_list &PropList) {
  return MImpl->complete_fusion(PropList);
}

} // namespace ext::codeplay::experimental
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
