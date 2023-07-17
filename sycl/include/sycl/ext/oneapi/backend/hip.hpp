//==--------- hip.hpp - SYCL HIP backend -----------------------------------==//
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

template <>
inline backend_return_t<backend::ext_oneapi_hip, device>
get_native<backend::ext_oneapi_hip, device>(const device &Obj)
{
  if (Obj.get_backend() != backend::ext_oneapi_hip)
  {
    throw sycl::exception(make_error_code(errc::backend_mismatch),
                          "Backends mismatch");
  }
  // HIP uses a 32-bit int instead of an opaque pointer like other backends,
  // so we need a specialization with static_cast instead of reinterpret_cast.
  return static_cast<backend_return_t<backend::ext_oneapi_hip, device>>(
      Obj.getNative());
}

template <>
inline device make_device<backend::ext_oneapi_hip>(
    const backend_input_t<backend::ext_oneapi_hip, device> &BackendObject)
{
  auto devs = device::get_devices(info::device_type::gpu);
  for (auto &dev : devs)
  {
    if (dev.get_backend() == backend::ext_oneapi_hip &&
        BackendObject == get_native<backend::ext_oneapi_hip>(dev))
    {
      return dev;
    }
  }
  pi_native_handle NativeHandle = static_cast<pi_native_handle>(BackendObject);
  return detail::make_device(NativeHandle,
                             backend::ext_oneapi_hip);
}

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
