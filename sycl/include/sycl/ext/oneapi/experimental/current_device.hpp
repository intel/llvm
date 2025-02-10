//==--------- current_device.hpp - sycl_ext_oneapi_current_device ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/device.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental::this_thread {

namespace detail {
thread_local sycl::device current_device = sycl::device{sycl::default_selector_v};
} // namespace detail

/// @return The current default device for the calling host thread. If
/// `set_current_device()` has not been called by this thread, returns the
/// device selected by the default device selector.
///
/// @pre The function is called from a host thread, executing outside of a host
/// task or an asynchronous error handler.
sycl::device get_current_device() { return detail::current_device; }

/// @brief Sets the current default device to `dev` for the calling host thread.
///
/// @pre The function is called from a host thread, executing outside of a host
/// task or an asynchronous error handler.
void set_current_device(sycl::device dev) { detail::current_device = dev; }

} // namespace ext::oneapi::experimental::this_thread
} // namespace _V1
} // namespace sycl
