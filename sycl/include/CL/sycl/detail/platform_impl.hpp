//==-------------- platform_impl.hpp - SYCL platform -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/force_device.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/detail/platform_info.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/stl.hpp>

namespace cl {
namespace sycl {

// Forward declaration
class device_selector;
class device;

namespace detail {

// TODO: implement extension management for host device
// TODO: implement parameters treatment for host device
class platform_impl {
public:
  platform_impl() : MHostPlatform(true) {};

  explicit platform_impl(const device_selector &DeviceSelector);

  explicit platform_impl(RT::PiPlatform Platform) : MPlatform(Platform) {}

  ~platform_impl() = default;

  bool has_extension(const string_class &ExtensionName) const {
    if (is_host())
      return false;

    string_class all_extension_names =
        get_platform_info<string_class, info::platform::extensions>::get(MPlatform);
    return (all_extension_names.find(ExtensionName) != std::string::npos);
  }

  vector_class<device>
  get_devices(info::device_type DeviceType = info::device_type::all) const;

  template <info::platform param>
  typename info::param_traits<info::platform, param>::return_type
  get_info() const {
    if (is_host())
      return get_platform_info_host<param>();

    return get_platform_info<
        typename info::param_traits<info::platform, param>::return_type,
        param>::get(this->getHandleRef());
  }

  bool is_host() const { return MHostPlatform; };

  cl_platform_id get() const { return pi::cast<cl_platform_id>(MPlatform); }

  // Returns underlying native platform object.
  const RT::PiPlatform &getHandleRef() const {
    if (is_host())
      throw invalid_object_error("This instance of platform is a host instance");

    return MPlatform;
  }

  static vector_class<platform> get_platforms();

private:
  bool MHostPlatform = false;
  RT::PiPlatform MPlatform = 0;
};
} // namespace detail
} // namespace sycl
} // namespace cl
