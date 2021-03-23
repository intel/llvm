//==-------------- backend_traits.hpp - SYCL backend traits ----------------==//
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
template <backend Backend> class backend_traits {
public:
  template <class T> using input_type = typename interop<Backend, T>::type;

  template <class T> using return_type = typename interop<Backend, T>::type;

  // TODO define errc once SYCL2020-style exceptions are supported.
};
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
