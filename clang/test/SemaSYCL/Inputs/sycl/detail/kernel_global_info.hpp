//==-------------------- kernel_global_info.hpp -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/export.hpp>

namespace {
namespace free_function_info_map {

__SYCL_EXPORT void add(const void *DeviceGlobalPtr, const char *UniqueId);
__SYCL_EXPORT void remove(const void *DeviceGlobalPtr, const char *UniqueId);

} // namespace free_function_info_map
}
