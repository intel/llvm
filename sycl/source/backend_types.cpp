//==------------------- backend_types.cpp ----------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/backend_types.hpp>

#include <ostream>
#include <sycl/detail/export.hpp>

namespace sycl {
inline namespace _V1 {

__SYCL_EXPORT std::ostream &operator<<(std::ostream &Out, backend be) {
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
  case backend::ext_intel_esimd_emulator:
    Out << "ext_intel_esimd_emulator";
    break;
  case backend::ext_oneapi_hip:
    Out << "ext_oneapi_hip";
    break;
  case backend::ext_native_cpu:
    Out << "ext_native_cpu";
    break;
  case backend::all:
    Out << "all";
  }
  return Out;
}

} // namespace _V1
} // namespace sycl