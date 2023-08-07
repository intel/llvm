//==--------- handler_proxy.hpp - Proxy methods to call in handler ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp> // for image_target, target
#include <sycl/detail/export.hpp> // for __SYCL_EXPORT

namespace sycl {
inline namespace _V1 {

class handler;

namespace detail {

class AccessorBaseHost;
class UnsampledImageAccessorBaseHost;
class SampledImageAccessorBaseHost;

#ifdef __SYCL_DEVICE_ONLY__
// In device compilation accessor isn't inherited from host base classes, so
// can't detect by it. Since we don't expect it to be ever called in device
// execution, just use blind void *.
inline void associateWithHandler(handler &, void *, access::target) {}
inline void associateWithHandler(handler &, void *, image_target) {}
#else
__SYCL_EXPORT void associateWithHandler(handler &, AccessorBaseHost *,
                                        access::target);
__SYCL_EXPORT void
associateWithHandler(handler &, UnsampledImageAccessorBaseHost *, image_target);
__SYCL_EXPORT void
associateWithHandler(handler &, SampledImageAccessorBaseHost *, image_target);
#endif
} // namespace detail
} // namespace _V1
} // namespace sycl
