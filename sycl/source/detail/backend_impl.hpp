//==------------------ backend_impl.hpp - get impls backend
//-------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/backend_types.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

template <class T> backend getImplBackend(const T &Impl) {
  backend Result;
  if (Impl->is_host())
    Result = backend::host;
  else
    Result = Impl->getPlugin().getBackend();

  return Result;
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
