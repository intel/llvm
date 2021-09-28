//==---------------- handler_impl.hpp - SYCL handler -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/kernel_bundle_impl.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

using KernelBundleImplPtr = std::shared_ptr<detail::kernel_bundle_impl>;

enum class HandlerSubmissionState : std::uint8_t {
  NO_STATE = 0,
  EXPLICIT_KERNEL_BUNDLE_STATE,
  SPEC_CONST_SET_STATE,
};

class handler_impl {
public:
  handler_impl() = default;

  /// Registers mutually exclusive submission states.
  HandlerSubmissionState MSubmissionState =
      detail::HandlerSubmissionState::NO_STATE;
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
