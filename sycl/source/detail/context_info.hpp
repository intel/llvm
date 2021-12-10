//==---------------- context_info.hpp - SYCL context -----------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <detail/context_impl.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
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

// Specialization for atomic_memory_order_capabilities, PI returns a bitfield
template <>
struct get_context_info<info::context::atomic_memory_order_capabilities> {
  using RetType = typename info::param_traits<
      info::context,
      info::context::atomic_memory_order_capabilities>::return_type;

  static RetType get(RT::PiContext ctx, const plugin &Plugin) {
    pi_memory_order_capabilities Result;
    Plugin.call<PiApiKind::piContextGetInfo>(
        ctx,
        pi::cast<pi_context_info>(
            info::context::atomic_memory_order_capabilities),
        sizeof(Result), &Result, nullptr);
    return readMemoryOrderBitfield(Result);
  }
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
