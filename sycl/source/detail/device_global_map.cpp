//==-------------------- device_global_map.cpp -----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/program_manager/program_manager.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail::device_global_map {

__SYCL_EXPORT void add(const void *DeviceGlobalPtr, const char *UniqueId) {
  detail::ProgramManager::getInstance().addOrInitDeviceGlobalEntry(
      DeviceGlobalPtr, UniqueId);
}

} // namespace detail::device_global_map
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
