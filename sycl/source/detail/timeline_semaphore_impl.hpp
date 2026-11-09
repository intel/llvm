//==-- timeline_semaphore_impl.hpp --- SYCL timeline semaphore impl --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/context_impl.hpp>
#include <sycl/context.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace sycl {
inline namespace _V1 {
namespace detail {

class queue_impl;

class timeline_semaphore_impl {
public:
  timeline_semaphore_impl(const std::shared_ptr<context_impl> &Context,
                          uint64_t InitialValue);
  ~timeline_semaphore_impl();

  timeline_semaphore_impl(const timeline_semaphore_impl &) = delete;
  timeline_semaphore_impl &operator=(const timeline_semaphore_impl &) = delete;

  context get_context() const;
  context_impl &getContextImpl() const { return *MContext; }

  void wait(uint64_t Value) const;
  void signal(uint64_t Value) const;

  /// Throws errc::invalid if \p Value is larger than the largest value that
  /// the semaphore may hold on every device of the context. This applies to
  /// initial, signal and wait values alike.
  static void checkValue(context_impl &Context, uint64_t Value);
  void checkValue(uint64_t Value) const { checkValue(*MContext, Value); }

  /// Enqueues a signal or wait operation to \p Queue after \p RawEvents.
  ///
  /// On an out-of-order queue the operation is surrounded by barriers, so that
  /// it executes after all commands submitted before and gates all commands
  /// submitted after it. \p OutEvent then refers to the closing barrier.
  ur_result_t enqueue(queue_impl &Queue, uint64_t Value, bool IsSignal,
                      const std::vector<ur_event_handle_t> &RawEvents,
                      ur_event_handle_t *OutEvent) const;

private:
  std::shared_ptr<context_impl> MContext;
  ur_exp_timeline_semaphore_handle_t MSemaphore = nullptr;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
