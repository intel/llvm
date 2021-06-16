//==---------------- context_info.hpp - SYCL context -----------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <sycl/__impl/detail/common.hpp>
#include <sycl/__impl/info/info_desc.hpp>
#include <detail/context_impl.hpp>

namespace __sycl_internal {
inline namespace __v1 {
namespace detail {

template <info::context param> struct get_context_info {
  using RetType =
      typename info::param_traits<info::context, param>::return_type;

  static RetType get(RT::PiContext ctx, const plugin &Plugin) {
    RetType Result = 0;
    // TODO catch an exception and put it to list of asynchronous exceptions
    Plugin.call<PiApiKind::piContextGetInfo>(ctx,
                                             pi::cast<pi_context_info>(param),
                                             sizeof(Result), &Result, nullptr);
    return Result;
  }
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
