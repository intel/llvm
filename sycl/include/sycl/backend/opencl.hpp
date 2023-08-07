//==---------------- opencl.hpp - SYCL OpenCL backend ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/backend_types.hpp>             // for backend
#include <sycl/context.hpp>                   // for context
#include <sycl/detail/backend_traits.hpp>     // for interop
#include <sycl/detail/defines_elementary.hpp> // for __SYCL_DEPRECATED
#include <sycl/detail/export.hpp>             // for __SYCL_EXPORT
#include <sycl/detail/pi.h>                   // for pi_native_handle
#include <sycl/device.hpp>                    // for device
#include <sycl/platform.hpp>                  // for platform
#include <sycl/queue.hpp>                     // for queue

#include <string>      // for string
#include <type_traits> // for enable_if_t

namespace sycl {
inline namespace _V1 {
namespace opencl {
// Implementation of various "make" functions resides in SYCL RT because
// creating SYCL objects requires knowing details not accessible here.
// Note that they take opaque pi_native_handle that real OpenCL handles
// are casted to.
//
__SYCL_EXPORT platform make_platform(pi_native_handle NativeHandle);
__SYCL_EXPORT device make_device(pi_native_handle NativeHandle);
__SYCL_EXPORT context make_context(pi_native_handle NativeHandle);
__SYCL_EXPORT queue make_queue(const context &Context,
                               pi_native_handle InteropHandle);

__SYCL_EXPORT bool has_extension(const sycl::platform &SyclPlatform,
                                 const std::string &Extension);
__SYCL_EXPORT bool has_extension(const sycl::device &SyclDevice,
                                 const std::string &Extension);

// Construction of SYCL platform.
template <typename T,
          typename std::enable_if_t<std::is_same_v<T, platform>> * = nullptr>
__SYCL_DEPRECATED("Use SYCL 2020 sycl::make_platform free function")
T make(typename detail::interop<backend::opencl, T>::type Interop) {
  return make_platform(detail::pi::cast<pi_native_handle>(Interop));
}

// Construction of SYCL device.
template <typename T,
          typename std::enable_if_t<std::is_same_v<T, device>> * = nullptr>
__SYCL_DEPRECATED("Use SYCL 2020 sycl::make_device free function")
T make(typename detail::interop<backend::opencl, T>::type Interop) {
  return make_device(detail::pi::cast<pi_native_handle>(Interop));
}

// Construction of SYCL context.
template <typename T,
          typename std::enable_if_t<std::is_same_v<T, context>> * = nullptr>
__SYCL_DEPRECATED("Use SYCL 2020 sycl::make_context free function")
T make(typename detail::interop<backend::opencl, T>::type Interop) {
  return make_context(detail::pi::cast<pi_native_handle>(Interop));
}

// Construction of SYCL queue.
template <typename T,
          typename std::enable_if_t<std::is_same_v<T, queue>> * = nullptr>
__SYCL_DEPRECATED("Use SYCL 2020 sycl::make_queue free function")
T make(const context &Context,
       typename detail::interop<backend::opencl, T>::type Interop) {
  return make_queue(Context, detail::pi::cast<pi_native_handle>(Interop));
}
} // namespace opencl
} // namespace _V1
} // namespace sycl
