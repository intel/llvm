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
  /// Constructs platform_impl for a SYCL host platform.
  platform_impl() : MHostPlatform(true) {}

  /// Constructs platform_impl instance using device selector.
  ///
  /// One of the SYCL devices that is associated with the constructed SYCL
  /// platform instance must be the SYCL device that is produced from the
  /// provided device selector.
  ///
  /// @param DeviceSelector is an instance of SYCL device_selector.
  explicit platform_impl(const device_selector &DeviceSelector);

  /// Constructs platform_impl from a plug-in interoperability platform handle.
  ///
  /// @param Platform is a raw plug-in platform handle.
  explicit platform_impl(RT::PiPlatform Platform) : MPlatform(Platform) {}

  ~platform_impl() = default;

  /// Checks if this platform supports extension.
  ///
  /// @param ExtensionName is a string containing extension name.
  /// @return true if platform supports specified extension.
  bool has_extension(const string_class &ExtensionName) const {
    if (is_host())
      return false;

    string_class all_extension_names =
        get_platform_info<string_class, info::platform::extensions>::get(
            MPlatform);
    return (all_extension_names.find(ExtensionName) != std::string::npos);
  }

  /// Gets all SYCL devices associated with this platform.
  ///
  /// If this platform is a host platform, resulting vector contains only
  /// a single SYCL host device. If there are no devices that match given device
  /// type, resulting vector is empty.
  ///
  /// @param DeviceType is a SYCL device type.
  /// @return a vector of SYCL devices.
  vector_class<device>
  get_devices(info::device_type DeviceType = info::device_type::all) const;

  /// Queries this SYCL platform for info.
  ///
  /// The return type depends on information being queried.
  template <info::platform param>
  typename info::param_traits<info::platform, param>::return_type
  get_info() const {
    if (is_host())
      return get_platform_info_host<param>();

    return get_platform_info<
        typename info::param_traits<info::platform, param>::return_type,
        param>::get(this->getHandleRef());
  }

  /// Checks if this platform is a host platform.
  ///
  /// @return true if this SYCL platform is a host platform.
  bool is_host() const { return MHostPlatform; };

  /// Gets an OpenCL interoperability platform.
  ///
  /// @return an instance of OpenCL cl_platform_id.
  cl_platform_id get() const { return pi::cast<cl_platform_id>(MPlatform); }

  /// Gets raw underlying plug-in platform handle.
  ///
  /// Unlike get() method, this method does not retain handler. It is caller
  /// responsibility to make sure that platform stays alive while raw handle
  /// is in use.
  ///
  /// @return a raw plug-in platform handle.
  const RT::PiPlatform &getHandleRef() const {
    if (is_host())
      throw invalid_object_error(
          "This instance of platform is a host instance");

    return MPlatform;
  }

  /// Gets all available SYCL platforms in the system.
  ///
  /// The resulting vector always contains a single SYCL host platform instance.
  ///
  /// @return a vector of all available SYCL platforms.
  static vector_class<platform> get_platforms();

private:
  bool MHostPlatform = false;
  RT::PiPlatform MPlatform = 0;
};
} // namespace detail
} // namespace sycl
} // namespace cl
