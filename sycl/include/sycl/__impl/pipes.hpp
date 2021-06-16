//==---------------- pipes.hpp - SYCL pipes ------------*- C++ -*-----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <sycl/__impl/INTEL/pipes.hpp>

namespace __sycl_internal {
inline namespace __v1 {
template <class name, class dataT, int32_t min_capacity = 0>
using pipe = INTEL::pipe<name, dataT, min_capacity>;
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

namespace sycl {
  using namespace __sycl_internal::__v1;
}
