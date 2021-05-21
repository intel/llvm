//==-------------- memory_enums.hpp --- SYCL enums -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/ONEAPI/atomic_enums.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
using ONEAPI::memory_scope;

#if __cplusplus >= 201703L
inline constexpr auto memory_scope_work_item = memory_scope::work_item;
inline constexpr auto memory_scope_sub_group = memory_scope::sub_group;
inline constexpr auto memory_scope_work_group = memory_scope::work_group;
inline constexpr auto memory_scope_device = memory_scope::device;
inline constexpr auto memory_scope_system = memory_scope::system;
#endif
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
