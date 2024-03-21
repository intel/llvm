//==------------------ backend_impl.hpp - get impls backend ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <cassert>
#include <sycl/backend_types.hpp>
#include <sycl/event.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

template <class T> backend getImplBackend(const T &Impl) {
  // Experimental host task allows the user to get backend for event impls
  if constexpr (std::is_same_v<T, std::shared_ptr<event_impl>>) {
    assert((!Impl->is_host() || Impl->backendSet()) &&
           "interop_handle::add_native_events must be "
           "used in order for a host "
           "task event to have a native event");
  } else {
    assert(!Impl->is_host() && "Cannot get the backend for host.");
  }
  return Impl->getContextImplPtr()->getBackend();
}

} // namespace detail
} // namespace _V1
} // namespace sycl
