//==---------- ur.cpp - Unified Runtime integration helpers ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file
///
/// Implementation of C++ utilities for Unified Runtime integration.
///
/// \ingroup sycl_ur

#include <sycl/detail/ur.hpp>

#include <iostream>

namespace sycl {
inline namespace _V1 {
namespace detail {
namespace ur {

// Report error and no return (keeps compiler from printing warnings).
// TODO: Probably change that to throw a catchable exception,
//       but for now it is useful to see every failure.
//
[[noreturn]] void die(const char *Message) {
  std::cerr << "ur_die: " << Message << std::endl;
  std::terminate();
}

void assertion(bool Condition, const char *Message) {
  if (!Condition) {
    die(Message);
  }
}

} // namespace ur
} // namespace detail
} // namespace _V1
} // namespace sycl
