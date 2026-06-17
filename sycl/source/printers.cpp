//==------ printers.cpp - SYCL stream operator definitions ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Out-of-line definitions for stream operators (operator<<, operator>>) on
// SYCL types. Centralizing them here keeps <iostream>/<ostream>/<istream>
// out of the public headers on both host and device compilation paths;
// public headers only carry __SYCL_EXPORT declarations for the operators.
//
// Consumers that want to print SYCL values must include the standard
// stream header (typically <iostream>) themselves.
//
//===----------------------------------------------------------------------===//

#include <sycl/backend_types.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/half_type.hpp>

#include <detail/iostream_proxy.hpp>

namespace sycl {
inline namespace _V1 {

std::ostream &operator<<(std::ostream &Out, backend be) {
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
  return Out;
}

namespace detail::half_impl {

std::ostream &operator<<(std::ostream &O, sycl::half const &rhs) {
  O << static_cast<float>(rhs);
  return O;
}

std::istream &operator>>(std::istream &I, sycl::half &rhs) {
  float ValFloat = 0.0f;
  I >> ValFloat;
  rhs = ValFloat;
  return I;
}

} // namespace detail::half_impl

namespace ext::oneapi {

std::ostream &operator<<(std::ostream &O, bfloat16 const &rhs) {
  O << static_cast<float>(rhs);
  return O;
}

std::istream &operator>>(std::istream &I, bfloat16 &rhs) {
  float ValFloat = 0.0f;
  I >> ValFloat;
  rhs = ValFloat;
  return I;
}

} // namespace ext::oneapi

} // namespace _V1
} // namespace sycl
