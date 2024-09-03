//==--------- level_zero.cpp - SYCL Level-Zero backend ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/platform_impl.hpp>
#include <detail/plugin.hpp>
#include <detail/queue_impl.hpp>
#include <sycl/backend.hpp>
#include <sycl/sycl.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::level_zero::detail {
using namespace sycl::detail;

__SYCL_EXPORT device make_device(const platform &Platform,
                                 ur_native_handle_t NativeHandle) {
  const auto &Plugin = ur::getPlugin<backend::ext_oneapi_level_zero>();
  const auto &PlatformImpl = getSyclObjImpl(Platform);
  // Create UR device first.
  ur_device_handle_t UrDevice;
  Plugin->call<UrApiKind::urDeviceCreateWithNativeHandle>(
      NativeHandle, Plugin->getUrAdapter(), nullptr, &UrDevice);

  return detail::createSyclObjFromImpl<device>(
      PlatformImpl->getOrMakeDeviceImpl(UrDevice, PlatformImpl));
}

} // namespace ext::oneapi::level_zero::detail
} // namespace _V1
} // namespace sycl
