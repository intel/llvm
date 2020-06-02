//==--------- handler_proxy.hpp - Proxy methods to call in handler ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/access/access.hpp>
#include <CL/sycl/detail/export.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

class handler;

namespace detail {

class AccessorBaseHost;

__SYCL_EXPORT void associateWithHandler(handler &, AccessorBaseHost *,
                                        access::target);
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
