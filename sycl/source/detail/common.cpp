//==----------- common.cpp -----------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/common_info.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

const char *stringifyErrorCode(pi_int32 error) {
  switch (error) {
#define _PI_ERRC(NAME, VAL)                                                    \
  case NAME:                                                                   \
    return #NAME;
#include <CL/sycl/detail/pi_error.def>
#undef _PI_ERRC

  case PI_ERROR_FUNCTION_ADDRESS_IS_NOT_AVAILABLE:
    return "Function exists but address is not available";
  case PI_ERROR_PLUGIN_SPECIFIC_ERROR:
    return "The plugin has emitted a backend specific error";
  case PI_ERROR_COMMAND_EXECUTION_FAILURE:
    return "Command failed to enqueue/execute";
  default:
    return "Unknown OpenCL error code";
  }
}

std::vector<std::string> split_string(const std::string &str, char delimeter) {
  std::vector<std::string> result;
  size_t beg = 0;
  size_t length = 0;
  for (const auto &x : str) {
    if (x == delimeter) {
      result.push_back(str.substr(beg, length));
      beg += length + 1;
      length = 0;
      continue;
    }
    length++;
  }
  if (length != 0) {
    result.push_back(str.substr(beg, length));
  }
  return result;
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
