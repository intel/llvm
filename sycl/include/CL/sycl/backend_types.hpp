//==-------------- backend_types.hpp - SYCL backend types ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/defines.hpp>

#include <fstream>
#include <iostream>
#include <istream>
#include <string>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

enum class backend : char {
  host = 0,
  opencl = 1,
  level_zero = 2,
  cuda = 3,
  all = 4,
  esimd_cpu = 5,
  rocm = 6,
};

template <backend Backend, typename SYCLObjectT> struct interop;

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
  case backend::level_zero:
    Out << "level_zero";
    break;
  case backend::cuda:
    Out << "cuda";
    break;
  case backend::esimd_cpu:
    Out << "esimd_cpu";
    break;
  case backend::rocm:
    Out << "rocm";
    break;
  case backend::all:
    Out << "all";
  }
  return Out;
}

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
