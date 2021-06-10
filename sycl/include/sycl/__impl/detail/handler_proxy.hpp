//==--------- handler_proxy.hpp - Proxy methods to call in handler ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__impl/access/access.hpp>
#include <sycl/__impl/detail/export.hpp>

namespace __sycl_internal {
inline namespace __v1 {

class handler;

namespace detail {

class AccessorBaseHost;

__SYCL_EXPORT void associateWithHandler(handler &, AccessorBaseHost *,
                                        access::target);
} // namespace detail
} // namespace sycl
} // namespace __sycl_internal
