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
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/stl.hpp>
#include <detail/plugin.hpp>
#include <detail/platform_info.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
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

  /// Constructs platform_impl from a plug-in interoperability platform
  /// handle.
  ///
  /// @param Platform is a raw plug-in platform handle.
  explicit platform_impl(RT::PiPlatform APlatform, const plugin &APlugin)
      : MPlatform(APlatform), MPlugin(std::make_shared<plugin>(APlugin)) {}

  explicit platform_impl(RT::PiPlatform APlatform,
                         std::shared_ptr<plugin> APlugin)
      : MPlatform(APlatform), MPlugin(APlugin) {}

  ~platform_impl() = default;

  /// Checks if this platform supports extension.
  ///
  /// @param ExtensionName is a string containing extension name.
  /// @return true if platform supports specified extension.
  bool has_extension(const string_class &ExtensionName) const;

  /// Returns all SYCL devices associated with this platform.
  ///
  /// If this platform is a host platform and device type requested is either
  /// info::device_type::all or info::device_type::host, resulting vector
  /// contains only a single SYCL host device. If there are no devices that
  /// match given device type, resulting vector is empty.
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
  get_info() const;

  /// @return true if this SYCL platform is a host platform.
  bool is_host() const { return MHostPlatform; };

  bool is_cuda() const {
    const string_class CUDA_PLATFORM_STRING = "NVIDIA CUDA";
    const string_class PlatformName = get_platform_info<string_class, 
                                                  info::platform::name>::get(MPlatform, getPlugin());
    return PlatformName == CUDA_PLATFORM_STRING;
  }

  /// @return an instance of OpenCL cl_platform_id.
  cl_platform_id get() const {
    if (is_host())
      throw invalid_object_error(
          "This instance of platform is a host instance");

    return pi::cast<cl_platform_id>(MPlatform);
  }

  /// Returns raw underlying plug-in platform handle.
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

  /// Returns all available SYCL platforms in the system.
  ///
  /// By default the resulting vector always contains a single SYCL host
  /// platform instance. There are means to override this behavior for testing
  /// purposes. See environment variables guide for up-to-date instructions.
  ///
  /// @return a vector of all available SYCL platforms.
  static vector_class<platform> get_platforms();

  // @return the Plugin associated with this platform.
  const plugin &getPlugin() const {
    assert(!MHostPlatform && "Plugin is not available for Host.");
    return *MPlugin;
  }

private:
  bool MHostPlatform = false;
  RT::PiPlatform MPlatform = 0;
  std::shared_ptr<plugin> MPlugin;
};
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
