//==-------------- backend_types.hpp - SYCL backend types ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/defines.hpp>

#include <string>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

enum class backend : char {
  host = 0,
  opencl = 1,
  ext_oneapi_level_zero = 2,
  level_zero __SYCL2020_DEPRECATED("use 'ext_oneapi_level_zero' instead") =
      ext_oneapi_level_zero,
  ext_oneapi_cuda = 3,
  cuda __SYCL2020_DEPRECATED("use 'ext_oneapi_cuda' instead") = ext_oneapi_cuda,
  all = 4,
  ext_intel_esimd_emulator = 5,
  esimd_cpu __SYCL2020_DEPRECATED("use 'ext_oneapi_esimd_emulator' instead") =
      ext_intel_esimd_emulator,
  ext_oneapi_hip = 6,
  hip __SYCL2020_DEPRECATED("use 'ext_oneapi_hip' instead") = ext_oneapi_hip
};

template <backend Backend> class backend_traits;

template <backend Backend, typename SYCLObjectT>
using backend_input_t =
    typename backend_traits<Backend>::template input_type<SYCLObjectT>;
template <backend Backend, typename SYCLObjectT>
using backend_return_t =
    typename backend_traits<Backend>::template return_type<SYCLObjectT>;
    
inline std::string backend_to_string(const backend& be) {

  switch (be) {
  case backend::host:
    return "host";
  case backend::opencl:
    return "opencl";
  case backend::ext_oneapi_level_zero:
    return "ext_oneapi_level_zero";
  case backend::ext_oneapi_cuda:
    return "ext_oneapi_cuda";
  case backend::ext_intel_esimd_emulator:
    return "ext_intel_esimd_emulator";
  case backend::ext_oneapi_hip:
    return "ext_oneapi_hip";
  case backend::all:
    return "all";
  }
  return "";
}


} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
