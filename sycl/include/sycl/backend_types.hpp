//==-------------- backend_types.hpp - SYCL backend types ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines.hpp>
#include <sycl/detail/iostream_proxy.hpp>

#include <fstream>
#include <istream>
#include <string>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

enum class backend : char {
  host __SYCL2020_DEPRECATED("'host' backend is no longer supported") = 0,
  opencl = 1,
  ext_oneapi_level_zero = 2,
  level_zero __SYCL2020_DEPRECATED("use 'ext_oneapi_level_zero' instead") =
      ext_oneapi_level_zero,
  ext_oneapi_cuda = 3,
  cuda __SYCL2020_DEPRECATED("use 'ext_oneapi_cuda' instead") = ext_oneapi_cuda,
  all = 4,
  ext_intel_esimd_emulator = 5,
  esimd_cpu __SYCL2020_DEPRECATED("use 'ext_intel_esimd_emulator' instead") =
      ext_intel_esimd_emulator,
  ext_oneapi_hip = 6,
  hip __SYCL2020_DEPRECATED("use 'ext_oneapi_hip' instead") = ext_oneapi_hip,
  ext_oneapi_unified_runtime = 7,
};

template <backend Backend> class backend_traits;

template <backend Backend, typename SYCLObjectT>
using backend_input_t =
    typename backend_traits<Backend>::template input_type<SYCLObjectT>;
template <backend Backend, typename SYCLObjectT>
using backend_return_t =
    typename backend_traits<Backend>::template return_type<SYCLObjectT>;

inline std::ostream &operator<<(std::ostream &Out, backend be) {
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
  case backend::ext_oneapi_unified_runtime:
    Out << "ext_oneapi_unified_runtime";
    break;
  case backend::all:
    Out << "all";
  }
  return Out;
}

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
