//==--------- cuda.hpp - SYCL CUDA backend ---------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/backend.hpp>
#include <sycl/context.hpp>
#include <sycl/ext/oneapi/experimental/backend/backend_traits_cuda.hpp>

#include <vector>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::oneapi::cuda {

// Implementation of ext_oneapi_cuda::make<device>
inline __SYCL_EXPORT device make_device(pi_native_handle NativeHandle) {
  return sycl::detail::make_device(NativeHandle, backend::ext_oneapi_cuda);
}

// Implementation of cuda::has_native_event
inline __SYCL_EXPORT bool has_native_event(event sycl_event) {
  if (sycl_event.get_backend() == backend::ext_oneapi_cuda)
    return get_native<backend::ext_oneapi_cuda>(sycl_event) != nullptr;

  return false;
}

} // namespace ext::oneapi::cuda

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

// CUDA device specialization
template <>
inline device make_device<backend::ext_oneapi_cuda>(
    const backend_input_t<backend::ext_oneapi_cuda, device> &BackendObject) {
  auto devs = device::get_devices(info::device_type::gpu);
  for (auto &dev : devs) {
    if (dev.get_backend() == backend::ext_oneapi_cuda &&
        BackendObject == get_native<backend::ext_oneapi_cuda>(dev)) {
      return dev;
    }
  }
  pi_native_handle NativeHandle = static_cast<pi_native_handle>(BackendObject);
  return ext::oneapi::cuda::make_device(NativeHandle);
}

// CUDA event specialization
template <>
inline event make_event<backend::ext_oneapi_cuda>(
    const backend_input_t<backend::ext_oneapi_cuda, event> &BackendObject,
    const context &TargetContext) {
  return detail::make_event(detail::pi::cast<pi_native_handle>(BackendObject),
                            TargetContext, true,
                            /*Backend*/ backend::ext_oneapi_cuda);
}

// CUDA queue specialization
template <>
inline queue make_queue<backend::ext_oneapi_cuda>(
    const backend_input_t<backend::ext_oneapi_cuda, queue> &BackendObject,
    const context &TargetContext, const async_handler Handler) {
  int32_t nativeHandleDesc = 0;
  const property_list &PropList{};
  return detail::make_queue(detail::pi::cast<pi_native_handle>(BackendObject),
                            nativeHandleDesc, TargetContext, nullptr, true,
                            PropList, Handler,
                            /*Backend*/ backend::ext_oneapi_cuda);
}

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
