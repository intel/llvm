//==------------ fusion_wrapper.cpp ----------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/ext/codeplay/experimental/fusion_wrapper.hpp>

#include <detail/queue_impl.hpp>
#include <detail/scheduler/scheduler.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace codeplay {
namespace experimental {

fusion_wrapper::fusion_wrapper(queue &Queue)
    : MQueue{sycl::detail::getSyclObjImpl(Queue)} {
  if (!Queue.ext_codeplay_supports_fusion()) {
    throw sycl::exception(
        sycl::errc::invalid,
        "Cannot wrap a queue for fusion which doesn't support fusion");
  }
}

queue fusion_wrapper::get_queue() const {
  return sycl::detail::createSyclObjFromImpl<sycl::queue>(MQueue);
}

bool fusion_wrapper::is_in_fusion_mode() const { return false; }

void fusion_wrapper::start_fusion() {
  throw sycl::exception(sycl::errc::feature_not_supported,
                        "Fusion not yet implemented");
}

void fusion_wrapper::cancel_fusion() {
  throw sycl::exception(sycl::errc::feature_not_supported,
                        "Fusion not yet implemented");
}

event fusion_wrapper::complete_fusion(const property_list &PropList) {
  (void)PropList;
  throw sycl::exception(sycl::errc::feature_not_supported,
                        "Fusion not yet implemented");
}

} // namespace experimental
} // namespace codeplay
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
