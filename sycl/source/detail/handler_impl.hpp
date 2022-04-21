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
  handler_impl(std::shared_ptr<queue_impl> SubmissionPrimaryQueue,
               std::shared_ptr<queue_impl> SubmissionSecondaryQueue)
      : MSubmissionPrimaryQueue(std::move(SubmissionPrimaryQueue)),
        MSubmissionSecondaryQueue(std::move(SubmissionSecondaryQueue)){};

  void setStateExplicitKernelBundle() {
    if (MSubmissionState == HandlerSubmissionState::SPEC_CONST_SET_STATE)
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Kernel bundle cannot be explicitly set after a specialization "
          "constant has been set");
    MSubmissionState = HandlerSubmissionState::EXPLICIT_KERNEL_BUNDLE_STATE;
  }

  void setStateSpecConstSet() {
    if (MSubmissionState ==
        HandlerSubmissionState::EXPLICIT_KERNEL_BUNDLE_STATE)
      throw sycl::exception(make_error_code(errc::invalid),
                            "Specialization constants cannot be set after "
                            "explicitly setting the used kernel bundle");
    MSubmissionState = HandlerSubmissionState::SPEC_CONST_SET_STATE;
  }

  bool isStateExplicitKernelBundle() const {
    return MSubmissionState ==
           HandlerSubmissionState::EXPLICIT_KERNEL_BUNDLE_STATE;
  }

  /// Registers mutually exclusive submission states.
  HandlerSubmissionState MSubmissionState = HandlerSubmissionState::NO_STATE;

  /// Shared pointer to the primary queue implementation. This is different from
  /// the queue associated with the handler if the corresponding submission is
  /// a fallback from a previous submission.
  std::shared_ptr<queue_impl> MSubmissionPrimaryQueue;

  /// Shared pointer to the secondary queue implementation. Nullptr if no
  /// secondary queue fallback was given in the associated submission. This is
  /// equal to the queue associated with the handler if the corresponding
  /// submission is a fallback from a previous submission.
  std::shared_ptr<queue_impl> MSubmissionSecondaryQueue;

  // Stores auxiliary resources used by internal operations.
  std::vector<std::shared_ptr<const void>> MAuxiliaryResources;
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
