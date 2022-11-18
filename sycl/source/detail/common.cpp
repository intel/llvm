//==----------- common.cpp -----------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/common.hpp>
#include <sycl/detail/common_info.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {
/// @brief CodeLocation information slot in thread local storage
/// @details This structure is maintained by the SYCL runtime to manage the
/// propagation of the code_location data down the stack without breaking ABI
/// compatibility
thread_local detail::code_location GCodeLocTLS = {};

const char *stringifyErrorCode(pi_int32 error) {
  switch (error) {
#define _PI_ERRC(NAME, VAL)                                                    \
  case NAME:                                                                   \
    return #NAME;
#define _PI_ERRC_WITH_MSG(NAME, VAL, MSG)                                      \
  case NAME:                                                                   \
    return MSG;
#include <sycl/detail/pi_error.def>
#undef _PI_ERRC
#undef _PI_ERRC_WITH_MSG

  default:
    return "Unknown error code";
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
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
