//==--------- cuda.hpp - SYCL CUDA backend ---------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/backend.hpp>
#include <CL/sycl/context.hpp>
#include <CL/sycl/interop_handle.hpp>

#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {
namespace cuda {

// Implementation of cuda::make<device>
inline __SYCL_EXPORT device make_device(pi_native_handle NativeHandle) {
  return sycl::detail::make_device(NativeHandle, backend::cuda);
}

// Implementation of cuda::make<platform>
inline __SYCL_EXPORT platform make_platform(pi_native_handle NativeHandle) {
  return sycl::detail::make_platform(NativeHandle, backend::cuda);
}

} // namespace cuda
} // namespace oneapi
} // namespace ext

// CUDA context specialization
template <>
inline auto get_native<backend::ext_oneapi_cuda, context>(const context &C)
    -> backend_return_t<backend::ext_oneapi_cuda, context> {
  // create a vector to be returned
  backend_return_t<backend::ext_oneapi_cuda, context> ret;

  // get the native CUDA context from the SYCL object
  auto native = reinterpret_cast<
      backend_return_t<backend::ext_oneapi_cuda, context>::value_type>(
      C.getNative());
  ret.push_back(native);

  return ret;
}

// Specialisation of non-free context get_native
template <>
inline backend_return_t<backend::ext_oneapi_cuda, context>
context::get_native<backend::ext_oneapi_cuda>() const {
  return sycl::get_native<backend::ext_oneapi_cuda, context>(*this);
}

// CUDA device specialization
template <>
inline device make_device<backend::ext_oneapi_cuda>(
    const backend_input_t<backend::ext_oneapi_cuda, device> &BackendObject) {
  pi_native_handle NativeHandle = static_cast<pi_native_handle>(BackendObject);
  return ext::oneapi::cuda::make_device(NativeHandle);
}

// CUDA platform specialization
template <>
inline auto get_native<backend::ext_oneapi_cuda, platform>(const platform &C)
    -> backend_return_t<backend::ext_oneapi_cuda, platform> {
  // get list of platform devices, and transfer to native platform type
  std::vector<device> platform_devices = C.get_devices();
  std::vector<CUdevice> native_devices(platform_devices.size());

  // Get the native CUdevice type for each device in platform
  for (unsigned int i = 0; i < platform_devices.size(); ++i)
    native_devices[i] =
        get_native<backend::ext_oneapi_cuda>(platform_devices[i]);

  return native_devices;
}

// Specialisation of non-free platform get_native
template <>
inline backend_return_t<backend::ext_oneapi_cuda, platform>
platform::get_native<backend::ext_oneapi_cuda>() const {
  return sycl::get_native<backend::ext_oneapi_cuda, platform>(*this);
}

template <>
inline platform make_platform<backend::ext_oneapi_cuda>(
    const backend_input_t<backend::ext_oneapi_cuda, platform> &BackendObject) {
  pi_native_handle NativeHandle =
      detail::pi::cast<pi_native_handle>(&BackendObject);
  return ext::oneapi::cuda::make_platform(NativeHandle);
}

// Specialisation of interop_handles get_native_context
template <>
inline backend_return_t<backend::ext_oneapi_cuda, context>
interop_handle::get_native_context<backend::ext_oneapi_cuda>() const {
#ifndef __SYCL_DEVICE_ONLY__
  return std::vector{reinterpret_cast<CUcontext>(getNativeContext())};
#else
  // we believe this won't be ever called on device side
  return {};
#endif
}

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
