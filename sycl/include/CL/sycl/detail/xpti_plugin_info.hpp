//==----------- xpti_plugin_info.hpp - Plugin info wrapper for XPTI --------==//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/pi.h>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
struct XPTIPluginInfo {
  uint8_t backend;  // ID of the backend, same as in sycl::backend.
  pi_plugin plugin; // Plugin, that was used to perform PI call.
  void *next;       // [Provisional] Pointer to the extended call function info.
};
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
