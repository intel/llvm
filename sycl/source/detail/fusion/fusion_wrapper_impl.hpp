//==-- fusion_wrapper_impl.hpp - SYCL wrapper for queue for kernel fusion --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/queue_impl.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {
class fusion_wrapper_impl {
public:
  explicit fusion_wrapper_impl(std::shared_ptr<detail::queue_impl> Queue);

  std::shared_ptr<detail::queue_impl> get_queue() const;

  bool is_in_fusion_mode() const;

  void start_fusion();

  void cancel_fusion();

  event complete_fusion(const property_list &propList = {});

private:
  std::shared_ptr<detail::queue_impl> MQueue;
};
} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
