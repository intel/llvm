//==-------------- platform_impl.hpp - SYCL platform -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/platform_info.hpp>
#include <detail/plugin.hpp>
#include <sycl/backend.hpp>
#include <sycl/backend_types.hpp>
#include <sycl/detail/cl.h>
#include <sycl/detail/common.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/info/info_desc.hpp>

namespace sycl {
inline namespace _V1 {

// Forward declaration
class device_selector;
class device;
enum class aspect;

namespace detail {
class device_impl;

// TODO: implement extension management for host device
// TODO: implement parameters treatment for host device
class platform_impl {
public:
  /// Constructs platform_impl for a SYCL host platform.
  platform_impl() : MHostPlatform(true) {}

  /// Constructs platform_impl from a plug-in interoperability platform
  /// handle.
  ///
  /// \param APlatform is a raw plug-in platform handle.
  /// \param APlugin is a plug-in handle.
  explicit platform_impl(sycl::detail::pi::PiPlatform APlatform,
                         const std::shared_ptr<plugin> &APlugin)
      : MPlatform(APlatform), MPlugin(APlugin) {

    // Find out backend of the platform
    sycl::detail::pi::PiPlatformBackend PiBackend;
    APlugin->call_nocheck<PiApiKind::piPlatformGetInfo>(
        APlatform, PI_EXT_PLATFORM_INFO_BACKEND,
        sizeof(sycl::detail::pi::PiPlatformBackend), &PiBackend, nullptr);
    MBackend = convertBackend(PiBackend);
  }

  ~platform_impl() = default;

  /// Checks if this platform supports extension.
  ///
  /// \param ExtensionName is a string containing extension name.
  /// \return true if platform supports specified extension.
  bool has_extension(const std::string &ExtensionName) const;

  /// Checks if this platform supports usm.
  /// Non opencl backends are assumed to support it.
  ///
  /// \return true if platform supports usm.
  bool supports_usm() const;

  /// Returns all SYCL devices associated with this platform.
  ///
  /// If this platform is a host platform and device type requested is either
  /// info::device_type::all or info::device_type::host, resulting vector
  /// contains only a single SYCL host device. If there are no devices that
  /// match given device type, resulting vector is empty.
  ///
  /// \param DeviceType is a SYCL device type.
  /// \return a vector of SYCL devices.
  std::vector<device>
  get_devices(info::device_type DeviceType = info::device_type::all) const;

  /// Queries this SYCL platform for info.
  ///
  /// The return type depends on information being queried.
  template <typename Param> typename Param::return_type get_info() const;

  /// \return true if this SYCL platform is a host platform.
  bool is_host() const { return MHostPlatform; };

  /// Returns the backend of this platform.
  backend getBackend(void) const { return MBackend; }

  /// Get backend option.
  void getBackendOption(const char *frontend_option,
                        const char **backend_option) const {
    const auto &Plugin = getPlugin();
    sycl::detail::pi::PiResult Err =
        Plugin->call_nocheck<PiApiKind::piPluginGetBackendOption>(
            MPlatform, frontend_option, backend_option);
    Plugin->checkPiResult(Err);
  }

  /// \return an instance of OpenCL cl_platform_id.
  cl_platform_id get() const {
    if (is_host()) {
      throw invalid_object_error(
          "This instance of platform doesn't support OpenCL interoperability.",
          PI_ERROR_INVALID_PLATFORM);
    }
    return pi::cast<cl_platform_id>(MPlatform);
  }

  /// Returns raw underlying plug-in platform handle.
  ///
  /// Unlike get() method, this method does not retain handler. It is caller
  /// responsibility to make sure that platform stays alive while raw handle
  /// is in use.
  ///
  /// \return a raw plug-in platform handle.
  const sycl::detail::pi::PiPlatform &getHandleRef() const {
    if (is_host())
      throw invalid_object_error("This instance of platform is a host instance",
                                 PI_ERROR_INVALID_PLATFORM);

    return MPlatform;
  }

  /// Returns all available SYCL platforms in the system.
  ///
  /// By default the resulting vector always contains a single SYCL host
  /// platform instance. There are means to override this behavior for testing
  /// purposes. See environment variables guide for up-to-date instructions.
  ///
  /// \return a vector of all available SYCL platforms.
  static std::vector<platform> get_platforms();

  // \return the Plugin associated with this platform.
  const PluginPtr &getPlugin() const {
    assert(!MHostPlatform && "Plugin is not available for Host.");
    return MPlugin;
  }

  /// Sets the platform implementation to use another plugin.
  ///
  /// \param PluginPtr is a pointer to a plugin instance
  /// \param Backend is the backend that we want this platform to use
  void setPlugin(PluginPtr &PluginPtr, backend Backend) {
    assert(!MHostPlatform && "Plugin is not available for Host");
    MPlugin = PluginPtr;
    MBackend = Backend;
  }

  /// Gets the native handle of the SYCL platform.
  ///
  /// \return a native handle.
  pi_native_handle getNative() const;

  /// Indicates if all of the SYCL devices on this platform have the
  /// given feature.
  ///
  /// \param Aspect is one of the values in Table 4.20 of the SYCL 2020
  /// Provisional Spec.
  ///
  /// \return true all of the SYCL devices on this platform have the
  /// given feature.
  bool has(aspect Aspect) const;

  /// Queries the device_impl cache to return a shared_ptr for the
  /// device_impl corresponding to the PiDevice.
  ///
  /// \param PiDevice is the PiDevice whose impl is requested
  ///
  /// \return a shared_ptr<device_impl> corresponding to the device
  std::shared_ptr<device_impl>
  getDeviceImpl(sycl::detail::pi::PiDevice PiDevice);

  /// Queries the device_impl cache to either return a shared_ptr
  /// for the device_impl corresponding to the PiDevice or add
  /// a new entry to the cache
  ///
  /// \param PiDevice is the PiDevice whose impl is requested
  ///
  /// \param PlatormImpl is the Platform for that Device
  ///
  /// \return a shared_ptr<device_impl> corresponding to the device
  std::shared_ptr<device_impl>
  getOrMakeDeviceImpl(sycl::detail::pi::PiDevice PiDevice,
                      const std::shared_ptr<platform_impl> &PlatformImpl);

  /// Static functions that help maintain platform uniquess and
  /// equality of comparison

  /// Returns the host platform impl
  ///
  /// \return the host platform impl
  static std::shared_ptr<platform_impl> getHostPlatformImpl();

  /// Queries the cache to see if the specified PiPlatform has been seen
  /// before.  If so, return the cached platform_impl, otherwise create a new
  /// one and cache it.
  ///
  /// \param PiPlatform is the PI Platform handle representing the platform
  /// \param Plugin is the PI plugin providing the backend for the platform
  /// \return the platform_impl representing the PI platform
  static std::shared_ptr<platform_impl>
  getOrMakePlatformImpl(sycl::detail::pi::PiPlatform PiPlatform,
                        const PluginPtr &Plugin);

  /// Queries the cache for the specified platform based on an input device.
  /// If found, returns the the cached platform_impl, otherwise creates a new
  /// one and caches it.
  ///
  /// \param PiDevice is the PI device handle for the device whose platform is
  /// desired
  /// \param Plugin is the PI plugin providing the backend for the device and
  /// platform
  /// \return the platform_impl that contains the input device
  static std::shared_ptr<platform_impl>
  getPlatformFromPiDevice(sycl::detail::pi::PiDevice PiDevice,
                          const PluginPtr &Plugin);

  // when getting sub-devices for ONEAPI_DEVICE_SELECTOR we may temporarily
  // ensure every device is a root one.
  bool MAlwaysRootDevice = false;

private:
  std::shared_ptr<device_impl>
  getDeviceImplHelper(sycl::detail::pi::PiDevice PiDevice);

  // Helper to filter reportable devices in the platform
  template <typename ListT, typename FilterT>
  std::vector<int>
  filterDeviceFilter(std::vector<sycl::detail::pi::PiDevice> &PiDevices,
                     ListT *FilterList) const;

  bool MHostPlatform = false;
  sycl::detail::pi::PiPlatform MPlatform = 0;
  backend MBackend;

  PluginPtr MPlugin;
  std::vector<std::weak_ptr<device_impl>> MDeviceCache;
  std::mutex MDeviceMapMutex;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
