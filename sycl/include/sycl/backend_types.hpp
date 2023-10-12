//==-------------- backend_types.hpp - SYCL backend types ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp> // for __SYCL2020_DEPRECATED

#include <iosfwd> // for operator<<, ostream
#include <sycl/detail/export.hpp>

namespace sycl {
inline namespace _V1 {

enum class backend : char {
  host __SYCL2020_DEPRECATED("'host' backend is no longer supported") = 0,
  opencl = 1,
  ext_oneapi_level_zero = 2,
  ext_oneapi_cuda = 3,
  all = 4,
  ext_intel_esimd_emulator __SYCL_DEPRECATED(
      "esimd emulator is no longer supported") = 5,
  ext_oneapi_hip = 6,
  ext_native_cpu = 7,
};

template <backend Backend> class backend_traits;

template <backend Backend, typename SYCLObjectT>
using backend_input_t =
    typename backend_traits<Backend>::template input_type<SYCLObjectT>;
template <backend Backend, typename SYCLObjectT>
using backend_return_t =
    typename backend_traits<Backend>::template return_type<SYCLObjectT>;

__SYCL_EXPORT std::ostream &operator<<(std::ostream &Out, backend be);

} // namespace _V1
} // namespace sycl
