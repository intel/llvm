//==---- cg_types.cpp - Auxiliary types required by command group class ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/cg_types.hpp>

namespace sycl {
inline namespace _V1 {

namespace detail {
__SYCL_EXPORT bool do_not_dce(void (*)(void *)) { return true; }
} // namespace detail
} // namespace _V1
} // namespace sycl
