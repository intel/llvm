//==----------- platform_impl.cpp ------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/platform_impl.hpp>
#include <CL/sycl/device.hpp>

namespace cl {
namespace sycl {
namespace detail {

vector_class<platform>
platform_impl_pi::get_platforms() {
  vector_class<platform> platforms;

  pi_uint32 num_platforms = 0;
  // TODO this bypasses PI_CALL macro as a temporary fix for the case with 0
  // OpenCL platforms
  RT::piPlatformsGet(0, 0, &num_platforms);
  info::device_type forced_type = detail::get_forced_type();

  if (num_platforms) {
    vector_class<RT::pi_platform> pi_platforms(num_platforms);
    PI_CALL(RT::piPlatformsGet(num_platforms, pi_platforms.data(), 0));

    for (pi_uint32 i = 0; i < num_platforms; i++) {

      platform plt =
        detail::createSyclObjFromImpl<platform>(
          std::make_shared<platform_impl_pi>(pi_platforms[i]));
      // Skip platforms which do not contain requested device types
      if (!plt.get_devices(forced_type).empty())
        platforms.push_back(plt);
    }
  }
  return platforms;
}

vector_class<device>
platform_impl_host::get_devices(info::device_type dev_type) const {
  vector_class<device> res;
  if (dev_type == info::device_type::host || dev_type == info::device_type::all)
    res.resize(1); // default device construct creates host device
  return res;
}

vector_class<device>
platform_impl_pi::get_devices(info::device_type deviceType) const {
  vector_class<device> res;
  if (deviceType == info::device_type::host)
    return res;

  pi_uint32 num_devices;
  auto err = RT::piDevicesGet(
    m_platform, pi_cast<pi_device_type>(deviceType), 0, 0, &num_devices);

  // TODO: remove this check when switched to PI as it would just return
  // zero in num_devices.
  if (err == CL_DEVICE_NOT_FOUND)
    return res;

  // TODO catch an exception and put it to list of asynchronous exceptions
  // TODO: remove dependency on CHECK_OCL_CODE
  CHECK_OCL_CODE(err);

  vector_class<RT::pi_device> pi_devices(num_devices);
  // TODO catch an exception and put it to list of asynchronous exceptions
  PI_CALL(RT::piDevicesGet(
    m_platform, pi_cast<pi_device_type>(deviceType), num_devices,
    pi_devices.data(), 0));

  std::for_each(pi_devices.begin(), pi_devices.end(),
                [&res](const RT::pi_device &a_pi_device) {
    device sycl_device =
      detail::createSyclObjFromImpl<device>(
        std::make_shared<device_impl_pi>(a_pi_device));
    res.push_back(sycl_device);
  });
  return res;
}

} // namespace detail
} // namespace sycl
} // namespace cl
