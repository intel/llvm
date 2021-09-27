//==-------------- backend_types.hpp - SYCL backend types ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/detail/export.hpp>

#include <iosfwd>
#include <string>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

enum class backend : char {
  host = 0,
  opencl = 1,
  ext_oneapi_level_zero = 2,
  level_zero __SYCL2020_DEPRECATED("use 'ext_oneapi_level_zero' instead") =
      ext_oneapi_level_zero,
  cuda = 3,
  all = 4,
  esimd_cpu = 5,
  hip = 6,
};

template <backend Backend, typename SYCLObjectT> struct interop;

template <backend Backend> class backend_traits;

template <backend Backend, typename SYCLObjectT>
using backend_input_t =
    typename backend_traits<Backend>::template input_type<SYCLObjectT>;
template <backend Backend, typename SYCLObjectT>
using backend_return_t =
    typename backend_traits<Backend>::template return_type<SYCLObjectT>;

__SYCL_EXPORT std::ostream &operator<<(std::ostream &Out, backend be);
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
