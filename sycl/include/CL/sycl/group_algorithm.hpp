//==----------- group_algorithm.hpp ------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/ONEAPI/group_algorithm.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
template <typename... Args>
auto reduce_over_group(Args &&... args)
    -> decltype(ONEAPI::reduce(std::forward<Args>(args)...)) {
  return ONEAPI::reduce(std::forward<Args>(args)...);
}
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
