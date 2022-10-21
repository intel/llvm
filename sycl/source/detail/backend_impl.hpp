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

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

template <class T> backend getImplBackend(const T &Impl) {
  // If that would ever become possible, event_impl::getNative needs to be
  // updated too.
  assert(!Impl->is_host() && "Cannot get the backend for host.");
  return Impl->getPlugin().getBackend();
}

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
