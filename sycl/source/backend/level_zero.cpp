//==--------- level_zero.cpp - SYCL Level-Zero backend ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/adapter_impl.hpp>
#include <detail/platform_impl.hpp>
#include <detail/queue_impl.hpp>
#include <detail/ur.hpp>
#include <sycl/backend_types.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::level_zero::detail {
using namespace sycl::detail;

__SYCL_EXPORT device make_device(const platform &Platform,
                                 ur_native_handle_t NativeHandle) {
  adapter_impl &Adapter = ur::getAdapter<backend::ext_oneapi_level_zero>();
  // Create UR device first.
  ur_device_handle_t UrDevice;
  Adapter.call<UrApiKind::urDeviceCreateWithNativeHandle>(
      NativeHandle, Adapter.getUrAdapter(), nullptr, &UrDevice);

  return detail::createSyclObjFromImpl<device>(
      getSyclObjImpl(Platform)->getOrMakeDeviceImpl(UrDevice));
}

} // namespace ext::oneapi::level_zero::detail
} // namespace _V1
} // namespace sycl
