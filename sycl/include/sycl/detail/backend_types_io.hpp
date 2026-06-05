//==----------- backend_types_io.hpp - SYCL backend stream support ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/backend_types.hpp>

#include <ostream>

namespace sycl {
inline namespace _V1 {

inline std::ostream &operator<<(std::ostream &Out, backend be) {
#if !defined(__SYCL_DEVICE_ONLY__)
  switch (be) {
  case backend::host:
    Out << "host";
    break;
  case backend::opencl:
    Out << "opencl";
    break;
  case backend::ext_oneapi_level_zero:
    Out << "ext_oneapi_level_zero";
    break;
  case backend::ext_oneapi_cuda:
    Out << "ext_oneapi_cuda";
    break;
  case backend::ext_oneapi_hip:
    Out << "ext_oneapi_hip";
    break;
  case backend::ext_oneapi_native_cpu:
    Out << "ext_oneapi_native_cpu";
    break;
  case backend::ext_oneapi_offload:
    Out << "ext_oneapi_offload";
    break;
  case backend::all:
    Out << "all";
  }
#endif // !defined(__SYCL_DEVICE_ONLY__)
  return Out;
}

} // namespace _V1
} // namespace sycl
