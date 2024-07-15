//==--------- hip.hpp - SYCL HIP backend -----------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/backend.hpp>
#include <sycl/detail/backend_traits_hip.hpp>

namespace sycl {
inline namespace _V1 {

template <>
inline backend_return_t<backend::ext_oneapi_hip, device>
get_native<backend::ext_oneapi_hip, device>(const device &Obj) {
  if (Obj.get_backend() != backend::ext_oneapi_hip) {
    throw exception(errc::backend_mismatch, "Backends mismatch");
  }
  // HIP uses a 32-bit int instead of an opaque pointer like other backends,
  // so we need a specialization with static_cast instead of reinterpret_cast.
  return static_cast<backend_return_t<backend::ext_oneapi_hip, device>>(
      Obj.getNative());
}

template <>
inline device make_device<backend::ext_oneapi_hip>(
    const backend_input_t<backend::ext_oneapi_hip, device> &BackendObject) {
  auto devs = device::get_devices(info::device_type::gpu);
  for (auto &dev : devs) {
    if (dev.get_backend() == backend::ext_oneapi_hip &&
        BackendObject == get_native<backend::ext_oneapi_hip>(dev)) {
      return dev;
    }
  }
  // The ext_oneapi_hip platform(s) adds all n available devices where n
  // is returned from call to `hipGetDeviceCount`.
  // Hence if this code is reached then the requested device ordinal must
  // not be visible to the driver.
  throw sycl::exception(make_error_code(errc::invalid),
                        "Native device has an invalid ordinal.");
}

} // namespace _V1
} // namespace sycl
