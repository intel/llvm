//==--------- handler_proxy.hpp - Proxy methods to call in handler ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>
#include <sycl/detail/export.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

class handler;

namespace detail {

class AccessorBaseHost;

#ifdef __SYCL_DEVICE_ONLY__
// In device compilation accessor isn't inherited from AccessorBaseHost, so
// can't detect by it. Since we don't expect it to be ever called in device
// execution, just use blind void *.
inline void associateWithHandler(handler &, void *, access::target) {}
#else
__SYCL_EXPORT void associateWithHandler(handler &, AccessorBaseHost *,
                                        access::target);
#endif
} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
