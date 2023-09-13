//==---- fusion_wrapper.hpp --- SYCL wrapper for queue for kernel fusion ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/export.hpp> // for __SYCL_EXPORT
#include <sycl/event.hpp>         // for event
#include <sycl/property_list.hpp> // for property_list
#include <sycl/queue.hpp>         // for queue

#include <memory> // for shared_ptr

namespace sycl {
inline namespace _V1 {

namespace detail {
class fusion_wrapper_impl;
}

namespace ext::codeplay::experimental {

///
/// A wrapper wrapping a sycl::queue to provide access to the kernel fusion API,
/// allowing to manage kernel fusion on the wrapped queue.
class __SYCL_EXPORT fusion_wrapper {

public:
  ///
  /// Wrap a queue to get access to the kernel fusion API.
  ///
  /// @throw sycl::exception with errc::invalid if trying to construct a wrapper
  /// on a queue which doesn't support fusion.
  explicit fusion_wrapper(queue &q);

  ///
  /// Access the queue wrapped by this fusion wrapper.
  queue get_queue() const;

  ///
  /// @brief Check whether the wrapped queue is in fusion mode or not.
  bool is_in_fusion_mode() const;

  ///
  /// @brief Set the wrapped queue into "fusion mode". This means that the
  /// kernels that are submitted in subsequent calls to queue::submit() are not
  /// submitted for execution right away, but rather added to a list of kernels
  /// that should be fused.
  ///
  /// @throw sycl::exception with errc::invalid if this operation is called on a
  /// queue which is already in fusion mode.
  void start_fusion();

  ///
  /// @brief Cancel the fusion and submit all kernels submitted since the last
  /// start_fusion() for immediate execution without fusion. The kernels are
  /// executed in the same order as they were initially submitted to the wrapped
  /// queue.
  ///
  /// This operation is asynchronous, i.e., it may return after the previously
  /// submitted kernels have been passed to the scheduler, but before any of the
  /// previously submitted kernel starts or completes execution. The events
  /// returned by submit() since the last call to start_fusion remain valid and
  /// can be used for synchronization.
  ///
  /// The queue is not in "fusion mode" anymore after this calls returns, until
  /// the next start_fusion().
  void cancel_fusion();

  ///
  /// @brief Complete the fusion: JIT-compile a fused kernel from all kernels
  /// submitted to the wrapped queue since the last start_fusion and submit the
  /// fused kernel for execution. Inside the fused kernel, the per-work-item
  /// effects are executed in the same order as the kernels were initially
  /// submitted.
  ///
  /// This operation is asynchronous, i.e., it may return after the JIT
  /// compilation is executed and the fused kernel is passed to the scheduler,
  /// but before the fused kernel starts or completes execution. The returned
  /// event allows to synchronize with the execution of the fused kernel. All
  /// events returned by queue::submit since the last call to start_fusion
  /// remain valid.
  ///
  /// The wrapped queue is not in "fusion mode" anymore after this calls
  /// returns, until the next start_fusion().
  ///
  /// @param properties Properties to take into account when performing fusion.
  event complete_fusion(const property_list &propList = {});

private:
  std::shared_ptr<detail::fusion_wrapper_impl> MImpl;
};
} // namespace ext::codeplay::experimental
} // namespace _V1
} // namespace sycl
