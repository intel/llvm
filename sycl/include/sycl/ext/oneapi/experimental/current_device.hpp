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
using namespace sycl::detail;
// Underlying `std::shared_ptr<device_impl>`'s lifetime is tied to the
// `global_handler`, so a subsequent `lock()` is expected to be successful when
// used from user app. We still go through `std::weak_ptr` here because our own
// unittests are linked statically against SYCL RT objects and have to implement
// some hacks to emulate the lifetime management done by the `global_handler`.
inline std::weak_ptr<device_impl> &get_current_device_impl() {
  static thread_local std::weak_ptr<device_impl> current_device{
      getSyclObjImpl(sycl::device{sycl::default_selector_v})};
  return current_device;
}
} // namespace detail

/// @return The current default device for the calling host thread. If
/// `set_current_device()` has not been called by this thread, returns the
/// device selected by the default device selector.
///
/// @pre The function is called from a host thread, executing outside of a host
/// task or an asynchronous error handler.
inline sycl::device get_current_device() {
  return detail::createSyclObjFromImpl<device>(
      detail::get_current_device_impl().lock());
}

/// @brief Sets the current default device to `dev` for the calling host thread.
///
/// @pre The function is called from a host thread, executing outside of a host
/// task or an asynchronous error handler.
inline void set_current_device(sycl::device dev) {
  detail::get_current_device_impl() = detail::getSyclObjImpl(dev);
}

} // namespace ext::oneapi::experimental::this_thread
} // namespace _V1
} // namespace sycl
