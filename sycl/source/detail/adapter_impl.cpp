//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definitions for the members of the adapter_impl
/// class.
///
//===----------------------------------------------------------------------===//

#include "adapter_impl.hpp"

namespace sycl {
inline namespace _V1 {
namespace detail {

void adapter_impl::ur_failed_throw_exception(sycl::errc errc,
                                             ur_result_t ur_result) const {
  assert(ur_result != UR_RESULT_SUCCESS);
  std::string message =
      __SYCL_UR_ERROR_REPORT(MBackend) + codeToString(ur_result);

  if (ur_result == UR_RESULT_ERROR_ADAPTER_SPECIFIC) {
    assert(!adapterReleased);
    const char *last_error_message = nullptr;
    int32_t adapter_error = 0;
    ur_result = call_nocheck<UrApiKind::urAdapterGetLastError>(
        MAdapter, &last_error_message, &adapter_error);
    if (last_error_message)
      message += "\n" + std::string(last_error_message) + "(adapter error )" +
                 std::to_string(adapter_error) + "\n";
  }

  throw set_ur_error(sycl::exception(sycl::make_error_code(errc), message),
                     ur_result);
}

} // namespace detail
} // namespace _V1
} // namespace sycl
