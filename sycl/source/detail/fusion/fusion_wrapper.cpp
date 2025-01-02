//==------------ fusion_wrapper.cpp ----------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO(#15184): Delete this file in the next ABI-breaking window.

#include <sycl/ext/codeplay/experimental/fusion_wrapper.hpp>

#include <detail/queue_impl.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::codeplay::experimental {

fusion_wrapper::fusion_wrapper(queue &Queue)
    : MQueue{detail::getSyclObjImpl(Queue)} {}

queue fusion_wrapper::get_queue() const {
  return detail::createSyclObjFromImpl<sycl::queue>(MQueue);
}

bool fusion_wrapper::is_in_fusion_mode() const { return false; }

void fusion_wrapper::start_fusion() {}

void fusion_wrapper::cancel_fusion() {}

event fusion_wrapper::complete_fusion(const property_list &PropList) {
  (void)PropList;
  throw sycl::exception(sycl::errc::feature_not_supported,
                        "Kernel fusion extension is no longer supported");
}

} // namespace ext::codeplay::experimental
} // namespace _V1
} // namespace sycl
