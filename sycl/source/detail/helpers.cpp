//==---------------- helpers.cpp - SYCL helpers ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/scheduler/commands.hpp>
#include <sycl/detail/helpers.hpp>

#include <detail/buffer_impl.hpp>
#include <detail/context_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/queue_impl.hpp>
#include <sycl/event.hpp>

#include <memory>

namespace sycl {
inline namespace _V1 {
using ContextImplPtr = std::shared_ptr<sycl::detail::context_impl>;
namespace detail {
void waitEvents(std::vector<sycl::event> DepEvents) {
  for (auto SyclEvent : DepEvents) {
    detail::getSyclObjImpl(SyclEvent)->waitInternal();
  }
}

void markBufferAsInternal(const std::shared_ptr<buffer_impl> &BufImpl) {
  BufImpl->markAsInternal();
}

} // namespace detail
} // namespace _V1
} // namespace sycl
