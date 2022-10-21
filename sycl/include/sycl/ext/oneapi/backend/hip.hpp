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
get_native<backend::ext_oneapi_hip, device>(const device &Obj) {
  // TODO use SYCL 2020 exception when implemented
  if (Obj.get_backend() != backend::ext_oneapi_hip) {
    throw sycl::runtime_error(errc::backend_mismatch, "Backends mismatch",
                              PI_ERROR_INVALID_OPERATION);
  }
  // HIP uses a 32-bit int instead of an opaque pointer like other backends,
  // so we need a specialization with static_cast instead of reinterpret_cast.
  return static_cast<backend_return_t<backend::ext_oneapi_hip, device>>(
      Obj.getNative());
}

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
