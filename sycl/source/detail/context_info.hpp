//==---------------- context_info.hpp - SYCL context -----------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <detail/context_impl.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/info_desc_helpers.hpp>
#include <sycl/info/info_desc.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

template <typename Param>
typename Param::return_type get_context_info(ur_context_handle_t Ctx,
                                             const PluginPtr &Plugin) {
  static_assert(is_context_info_desc<Param>::value,
                "Invalid context information descriptor");
  typename Param::return_type Result = 0;
  // TODO catch an exception and put it to list of asynchronous exceptions
  Plugin->call<UrApiKind::urContextGetInfo>(Ctx, UrInfoCode<Param>::value,
                                            sizeof(Result), &Result, nullptr);
  return Result;
}

} // namespace detail
} // namespace _V1
} // namespace sycl
