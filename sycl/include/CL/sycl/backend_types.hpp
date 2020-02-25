//==-------------- backend_types.hpp - SYCL backend types ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/detail/pi_sycl.hpp>

#include <fstream>
#include <iostream>
#include <istream>
#include <string>

namespace pi {
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
  case backend::all:
    Out << "all";
  }
  return Out;
}
} // namespace pi

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

using backend = pi::backend;

template <backend name, typename SYCLObjectT> struct interop;

inline std::ostream &operator<<(std::ostream &Out, backend be) {
  return pi::operator<<(Out, be);
}

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
