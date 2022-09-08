
//==---------------- opencl.hpp - SYCL OpenCL backend ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/backend.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
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

// Construction of SYCL platform.
template <typename T, typename detail::enable_if_t<
                          std::is_same<T, platform>::value> * = nullptr>
__SYCL_DEPRECATED("Use SYCL 2020 sycl::make_platform free function")
T make(typename detail::interop<backend::opencl, T>::type Interop) {
  return make_platform(detail::pi::cast<pi_native_handle>(Interop));
}

// Construction of SYCL device.
template <typename T, typename detail::enable_if_t<
                          std::is_same<T, device>::value> * = nullptr>
__SYCL_DEPRECATED("Use SYCL 2020 sycl::make_device free function")
T make(typename detail::interop<backend::opencl, T>::type Interop) {
  return make_device(detail::pi::cast<pi_native_handle>(Interop));
}

// Construction of SYCL context.
template <typename T, typename detail::enable_if_t<
                          std::is_same<T, context>::value> * = nullptr>
__SYCL_DEPRECATED("Use SYCL 2020 sycl::make_context free function")
T make(typename detail::interop<backend::opencl, T>::type Interop) {
  return make_context(detail::pi::cast<pi_native_handle>(Interop));
}

// Construction of SYCL queue.
template <typename T, typename detail::enable_if_t<
                          std::is_same<T, queue>::value> * = nullptr>
__SYCL_DEPRECATED("Use SYCL 2020 sycl::make_queue free function")
T make(const context &Context,
       typename detail::interop<backend::opencl, T>::type Interop) {
  return make_queue(Context, detail::pi::cast<pi_native_handle>(Interop));
}
} // namespace opencl
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
