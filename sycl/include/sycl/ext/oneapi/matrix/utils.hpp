//==------------------ utils.hpp - SYCL matrix ----------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <sycl/access/access.hpp>

namespace sycl {
namespace detail {
template <access::address_space Space, typename T> struct decorate_ptr {
  using type = std::conditional_t<
      Space == access::address_space::local_space,
      __attribute__((opencl_local)) T *,
      std::conditional_t<
          Space == access::address_space::global_space,
          __attribute__((opencl_global)) T *,
          std::conditional_t<Space == access::address_space::constant_space,
                             __attribute__((opencl_constant)) T *, T *>>>;
};
template <access::address_space Space, typename T>
using decorate_ptr_t = typename decorate_ptr<Space, T>::type;

} // namespace detail
} // namespace sycl