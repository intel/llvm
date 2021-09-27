//==---------- pi_helpers.hpp - Plugin Interface helpers -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/pi.hpp>

#include <sstream>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
namespace pi {

template <typename T>
void handleUnknownParamName(const char *functionName, T parameter) {
  std::stringstream stream;
  stream << "Unknown parameter " << parameter << " passed to " << functionName
         << "\n";
  auto str = stream.str();
  auto msg = str.c_str();
  die(msg);
}

// This macro is used to report invalid enumerators being passed to PI API
// GetInfo functions. It will print the name of the function that invoked it
// and the value of the unknown enumerator.
#define __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(parameter)                         \
  { cl::sycl::detail::pi::handleUnknownParamName(__func__, parameter); }

} // namespace pi

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
