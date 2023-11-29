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
inline namespace _V1 {
namespace detail {
/// @brief CodeLocation information slot in thread local storage
/// @details This structure is maintained by the SYCL runtime to manage the
/// propagation of the code_location data down the stack without breaking ABI
/// compatibility. This is used by the tls_code_loc_t class that is the
/// prescribed way to propagate the code location information
static thread_local detail::code_location GCodeLocTLS = {};

/// @brief Default constructor to use in lower levels of the calling stack to
/// check and see if code location object is available. If not, continue with
/// instrumentation as needed
tls_code_loc_t::tls_code_loc_t() {
  // Check TLS to see if a previously stashed code_location object is
  // available; if so, we are in a local scope.
  MLocalScope = GCodeLocTLS.fileName() && GCodeLocTLS.functionName();
}

/// @brief Constructor to use at the top level of the calling stack
/// @details This is usually a SYCL entry point used by the end user in their
/// application code. In this case, we still check to see if another code
/// location has been stashed in the TLS at a higher level. If not, we have the
/// code location information that must be active for the current calling scope.
tls_code_loc_t::tls_code_loc_t(const detail::code_location &CodeLoc) {
  // Check TLS to see if a previously stashed code_location object is
  // available; if so, then don't overwrite the previous information as we
  // are still in scope of the instrumented function
  MLocalScope = GCodeLocTLS.fileName() && GCodeLocTLS.functionName();

  if (!MLocalScope)
    // Update the TLS information with the code_location information
    GCodeLocTLS = CodeLoc;
}

/// @brief  If we are the top lovel scope,  reset the code location info
tls_code_loc_t::~tls_code_loc_t() {
  // Only reset the TLS data if the top level function is going out of scope
  if (!MLocalScope) {
    GCodeLocTLS = {};
  }
}

const detail::code_location &tls_code_loc_t::query() { return GCodeLocTLS; }

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
  std::vector<std::string> Result;
  size_t Start = 0;
  size_t End = 0;
  while ((End = str.find(delimeter, Start)) != std::string::npos) {
    Result.push_back(str.substr(Start, End - Start));
    Start = End + 1;
  }
  // Get the last substring and ignore the null character so we wouldn't get
  // double null characters \0\0 at the end of the substring
  End = str.find('\0');
  if (Start < End) {
    std::string LastSubStr(str.substr(Start, End - Start));
    // In case str has a delimeter at the end, the substring will be empty, so
    // we shouldn't add it to the final vector
    if (!LastSubStr.empty())
      Result.push_back(LastSubStr);
  }
  return Result;
}

} // namespace detail
} // namespace _V1
} // namespace sycl
