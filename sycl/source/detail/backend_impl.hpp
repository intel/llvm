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
inline namespace _V1 {
namespace detail {

template <class T> backend getImplBackend(const T &Impl) {
  return Impl->getContextImplPtr()->getBackend();
}

} // namespace detail
} // namespace _V1
} // namespace sycl
