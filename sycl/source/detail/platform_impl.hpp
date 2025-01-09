//==-------------- platform_impl.hpp - SYCL platform -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/adapter.hpp>
#include <detail/platform_info.hpp>
#include <detail/ur.hpp>
#include <detail/ur_info_code.hpp>
#include <sycl/backend_types.hpp>
#include <sycl/detail/cl.h>
#include <sycl/detail/common.hpp>
#include <sycl/detail/ur.hpp>
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
  /// Constructs platform_impl from a UR platform handle.
  ///
  /// \param APlatform is a raw plug-in platform handle.
  /// \param AAdapter is a plug-in handle.
  explicit platform_impl(ur_platform_handle_t APlatform,
                         const std::shared_ptr<Adapter> &AAdapter)
      : MPlatform(APlatform), MAdapter(AAdapter) {
    // Find out backend of the platform
    ur_platform_backend_t UrBackend = UR_PLATFORM_BACKEND_UNKNOWN;
    AAdapter->call_nocheck<UrApiKind::urPlatformGetInfo>(
        APlatform, UR_PLATFORM_INFO_BACKEND, sizeof(ur_platform_backend_t),
        &UrBackend, nullptr);
    MBackend = convertUrBackend(UrBackend);
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

  /// Queries this SYCL platform for SYCL backend-specific information.
  ///
  /// The return type depends on information being queried.
  template <typename Param>
  typename Param::return_type get_backend_info() const;

  /// Returns the backend of this platform.
  backend getBackend(void) const { return MBackend; }

  /// Get backend option.
  void getBackendOption(const char *frontend_option,
                        const char **backend_option) const {
    const auto &Adapter = getAdapter();
    ur_result_t Err =
        Adapter->call_nocheck<UrApiKind::urPlatformGetBackendOption>(
            MPlatform, frontend_option, backend_option);
    Adapter->checkUrResult(Err);
  }

  /// \return an instance of OpenCL cl_platform_id.
  cl_platform_id get() const {
    ur_native_handle_t nativeHandle = 0;
    getAdapter()->call<UrApiKind::urPlatformGetNativeHandle>(MPlatform,
                                                             &nativeHandle);
    return ur::cast<cl_platform_id>(nativeHandle);
  }

  /// Returns raw underlying UR platform handle.
  ///
  /// Unlike get() method, this method does not retain handler. It is caller
  /// responsibility to make sure that platform stays alive while raw handle
  /// is in use.
  ///
  /// \return a raw plug-in platform handle.
  const ur_platform_handle_t &getHandleRef() const { return MPlatform; }

  /// Returns all available SYCL platforms in the system.
  ///
  /// By default the resulting vector always contains a single SYCL host
  /// platform instance. There are means to override this behavior for testing
  /// purposes. See environment variables guide for up-to-date instructions.
  ///
  /// \return a vector of all available SYCL platforms.
  static std::vector<platform> get_platforms();

  // \return the Adapter associated with this platform.
  const AdapterPtr &getAdapter() const { return MAdapter; }

  /// Sets the platform implementation to use another adapter.
  ///
  /// \param AdapterPtr is a pointer to a adapter instance
  /// \param Backend is the backend that we want this platform to use
  void setAdapter(AdapterPtr &AdapterPtr, backend Backend) {
    MAdapter = AdapterPtr;
    MBackend = Backend;
  }

  /// Gets the native handle of the SYCL platform.
  ///
  /// \return a native handle.
  ur_native_handle_t getNative() const;

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
  /// device_impl corresponding to the UrDevice.
  ///
  /// \param UrDevice is the UrDevice whose impl is requested
  ///
  /// \return a shared_ptr<device_impl> corresponding to the device
  std::shared_ptr<device_impl> getDeviceImpl(ur_device_handle_t UrDevice);

  /// Queries the device_impl cache to either return a shared_ptr
  /// for the device_impl corresponding to the UrDevice or add
  /// a new entry to the cache
  ///
  /// \param UrDevice is the UrDevice whose impl is requested
  ///
  /// \param PlatormImpl is the Platform for that Device
  ///
  /// \return a shared_ptr<device_impl> corresponding to the device
  std::shared_ptr<device_impl>
  getOrMakeDeviceImpl(ur_device_handle_t UrDevice,
                      const std::shared_ptr<platform_impl> &PlatformImpl);

  /// Queries the cache to see if the specified UR platform has been seen
  /// before.  If so, return the cached platform_impl, otherwise create a new
  /// one and cache it.
  ///
  /// \param UrPlatform is the UR Platform handle representing the platform
  /// \param Adapter is the UR adapter providing the backend for the platform
  /// \return the platform_impl representing the UR platform
  static std::shared_ptr<platform_impl>
  getOrMakePlatformImpl(ur_platform_handle_t UrPlatform,
                        const AdapterPtr &Adapter);

  /// Queries the cache for the specified platform based on an input device.
  /// If found, returns the the cached platform_impl, otherwise creates a new
  /// one and caches it.
  ///
  /// \param UrDevice is the UR device handle for the device whose platform is
  /// desired
  /// \param Adapter is the UR adapter providing the backend for the device and
  /// platform
  /// \return the platform_impl that contains the input device
  static std::shared_ptr<platform_impl>
  getPlatformFromUrDevice(ur_device_handle_t UrDevice,
                          const AdapterPtr &Adapter);

  // when getting sub-devices for ONEAPI_DEVICE_SELECTOR we may temporarily
  // ensure every device is a root one.
  bool MAlwaysRootDevice = false;

private:
  std::shared_ptr<device_impl> getDeviceImplHelper(ur_device_handle_t UrDevice);

  // Helper to get the vector of platforms supported by a given UR adapter
  static std::vector<platform> getAdapterPlatforms(AdapterPtr &Adapter,
                                                   bool Supported = true);

  // Helper to filter reportable devices in the platform
  template <typename ListT, typename FilterT>
  std::vector<int>
  filterDeviceFilter(std::vector<ur_device_handle_t> &UrDevices,
                     ListT *FilterList) const;

  ur_platform_handle_t MPlatform = 0;
  backend MBackend;

  AdapterPtr MAdapter;

  std::vector<std::weak_ptr<device_impl>> MDeviceCache;
  std::mutex MDeviceMapMutex;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
