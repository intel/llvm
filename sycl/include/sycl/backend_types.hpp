//==-------------- backend_types.hpp - SYCL backend types ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp> // for __SYCL2020_DEPRECATED
#include <sycl/detail/export.hpp>             // for __SYCL_EXPORT

#include <iosfwd>      // for ostream
#include <string_view> // for string_view

namespace sycl {
inline namespace _V1 {

enum class backend : char {
  host __SYCL2020_DEPRECATED("'host' backend is no longer supported") = 0,
  opencl = 1,
  ext_oneapi_level_zero = 2,
  ext_oneapi_cuda = 3,
  all = 4,
  // No support anymore:
  // ext_intel_esimd_emulator  = 5,
  ext_oneapi_hip = 6,
  ext_oneapi_native_cpu = 7,
  ext_oneapi_offload = 8,
};

template <backend Backend> class backend_traits;

template <backend Backend, typename SYCLObjectT>
using backend_input_t =
    typename backend_traits<Backend>::template input_type<SYCLObjectT>;
template <backend Backend, typename SYCLObjectT>
using backend_return_t =
    typename backend_traits<Backend>::template return_type<SYCLObjectT>;

__SYCL_EXPORT std::ostream &operator<<(std::ostream &Out, backend be);

namespace detail {
inline std::string_view get_backend_name_no_vendor(backend Backend) {
  switch (Backend) {
  case backend::host:
    return "host";
  case backend::opencl:
    return "opencl";
  case backend::ext_oneapi_level_zero:
    return "level_zero";
  case backend::ext_oneapi_cuda:
    return "cuda";
  case backend::ext_oneapi_hip:
    return "hip";
  case backend::ext_oneapi_native_cpu:
    return "native_cpu";
  case backend::ext_oneapi_offload:
    return "offload";
  case backend::all:
    return "all";
  }

  return "";
}
} // namespace detail

} // namespace _V1
} // namespace sycl
