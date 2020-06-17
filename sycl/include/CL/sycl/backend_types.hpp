//==-------------- backend_types.hpp - SYCL backend types ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/defines.hpp>

#include <iostream>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

enum class backend : char { host, opencl, level0, cuda };

template <backend name, typename SYCLObjectT> struct interop;

inline std::ostream &operator<<(std::ostream &Out, backend be) {
  switch (be) {
  case backend::host:
    Out << std::string("host");
    break;
  case backend::opencl:
    Out << std::string("opencl");
    break;
  case backend::level0:
    Out << std::string("level0");
    break;
  case backend::cuda:
    Out << std::string("cuda");
    break;
  default:
    Out << std::string("unknown");
  }
  return Out;
}

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
