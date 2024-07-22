//==----------- common.cpp -----------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/common.hpp>

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

} // namespace detail
} // namespace _V1
} // namespace sycl
