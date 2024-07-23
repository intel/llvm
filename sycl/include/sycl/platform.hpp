//==---------------- platform.hpp - SYCL platform --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/aspects.hpp>
#include <sycl/backend_types.hpp>
#include <sycl/context.hpp>
#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/detail/info_desc_helpers.hpp>
#include <sycl/detail/owner_less_base.hpp>
#include <sycl/detail/pi.h>
#include <sycl/detail/string.hpp>
#include <sycl/detail/string_view.hpp>
#include <sycl/detail/util.hpp>
#include <sycl/device_selector.hpp>
#include <sycl/info/info_desc.hpp>

#ifdef __SYCL_INTERNAL_API
#include <sycl/detail/cl.h>
#endif

#include <cstddef>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace sycl {
inline namespace _V1 {
// TODO: make code thread-safe

// Forward declaration
class device;
class context;

template <backend BackendName, class SyclObjectT>
auto get_native(const SyclObjectT &Obj)
    -> backend_return_t<BackendName, SyclObjectT>;
namespace detail {
class platform_impl;

/// Allows to enable/disable "Default Context" extension
///
/// This API is in detail:: namespace because it's never supposed
/// to be called by end-user. It's necessary for internal use of
/// oneAPI components
///
/// \param Val Indicates if extension should be enabled/disabled
void __SYCL_EXPORT enable_ext_oneapi_default_context(bool Val);

template <typename ParamT> auto convert_to_abi_neutral(ParamT &&Info) {
  using ParamNoRef = std::remove_reference_t<ParamT>;
  if constexpr (std::is_same_v<ParamNoRef, std::string>) {
    return detail::string{Info};
  } else if constexpr (std::is_same_v<ParamNoRef, std::vector<std::string>>) {
    std::vector<detail::string> Res;
    Res.reserve(Info.size());
    for (std::string &Str : Info) {
      Res.push_back(detail::string{Str});
    }
    return Res;
  } else {
    return std::forward<ParamT>(Info);
  }
}

template <typename ParamT> auto convert_from_abi_neutral(ParamT &&Info) {
  using ParamNoRef = std::remove_reference_t<ParamT>;
  if constexpr (std::is_same_v<ParamNoRef, detail::string>) {
    return Info.c_str();
  } else if constexpr (std::is_same_v<ParamNoRef,
                                      std::vector<detail::string>>) {
    std::vector<std::string> Res;
    Res.reserve(Info.size());
    for (detail::string &Str : Info) {
      Res.push_back(Str.c_str());
    }
    return Res;
  } else {
    return std::forward<ParamT>(Info);
  }
}
} // namespace detail
namespace ext::oneapi {
// Forward declaration
class filter_selector;
} // namespace ext::oneapi

/// Encapsulates a SYCL platform on which kernels may be executed.
///
/// \ingroup sycl_api
class __SYCL_EXPORT platform : public detail::OwnerLessBase<platform> {
public:
  /// Constructs a SYCL platform using the default device.
  platform();

  /// Constructs a SYCL platform instance from an OpenCL cl_platform_id.
  ///
  /// The provided OpenCL platform handle is retained on SYCL platform
  /// construction.
  ///
  /// \param PlatformId is an OpenCL cl_platform_id instance.
#ifdef __SYCL_INTERNAL_API
  explicit platform(cl_platform_id PlatformId);
#endif

  /// Constructs a SYCL platform instance using a device_selector.
  ///
  /// One of the SYCL devices that is associated with the constructed SYCL
  /// platform instance must be the SYCL device that is produced from the
  /// provided device selector.
  ///
  /// \param DeviceSelector is an instance of a SYCL 1.2.1 device_selector
  __SYCL2020_DEPRECATED("SYCL 1.2.1 device selectors are deprecated. Please "
                        "use SYCL 2020 device selectors instead.")
  explicit platform(const device_selector &DeviceSelector);

  /// Constructs a SYCL platform instance using the platform of the device
  /// identified by the device selector provided.
  /// \param DeviceSelector is SYCL 2020 Device Selector, a simple callable that
  /// takes a device and returns an int
  template <typename DeviceSelector,
            typename =
                detail::EnableIfSYCL2020DeviceSelectorInvocable<DeviceSelector>>
  explicit platform(const DeviceSelector &deviceSelector)
      : platform(detail::select_device(deviceSelector)) {}

  platform(const platform &rhs) = default;

  platform(platform &&rhs) = default;

  platform &operator=(const platform &rhs) = default;

  platform &operator=(platform &&rhs) = default;

  bool operator==(const platform &rhs) const { return impl == rhs.impl; }

  bool operator!=(const platform &rhs) const { return !(*this == rhs); }

  /// Returns an OpenCL interoperability platform.
  ///
  /// \return an instance of OpenCL cl_platform_id.
#ifdef __SYCL_INTERNAL_API
  cl_platform_id get() const;
#endif

  /// Checks if platform supports specified extension.
  ///
  /// \param ExtensionName is a string containing extension name.
  /// \return true if specified extension is supported by this SYCL platform.
  __SYCL2020_DEPRECATED(
      "use platform::has() function with aspects APIs instead")
  bool has_extension(const std::string &ExtensionName) const {
    return has_extension(detail::string_view{ExtensionName});
  }

  /// Returns all SYCL devices associated with this platform.
  ///
  /// If this SYCL platform is a host platform, resulting vector contains only
  /// a single SYCL host device. If there are no devices that match given device
  /// type, resulting vector is empty.
  ///
  /// \param DeviceType is a SYCL device type.
  /// \return a vector of SYCL devices.
  std::vector<device>
  get_devices(info::device_type DeviceType = info::device_type::all) const;

  /// Queries this SYCL platform for info.
  ///
  /// The return type depends on information being queried.
  template <typename Param>
  typename detail::is_platform_info_desc<Param>::return_type get_info() const {
    return detail::convert_from_abi_neutral(get_info_impl<Param>());
  }

  /// Queries this SYCL platform for SYCL backend-specific info.
  ///
  /// The return type depends on information being queried.
  template <typename Param>
  typename detail::is_backend_info_desc<Param>::return_type
  get_backend_info() const;

  /// Returns all available SYCL platforms in the system.
  ///
  /// The resulting vector always contains a single SYCL host platform instance.
  ///
  /// \return a vector of all available SYCL platforms.
  static std::vector<platform> get_platforms();

  /// Returns the backend associated with this platform.
  ///
  /// \return the backend associated with this platform
  backend get_backend() const noexcept;

// Clang may warn about the use of diagnose_if in __SYCL_WARN_IMAGE_ASPECT, so
// we disable that warning as we make appropriate checks to ensure its
// existence.
// TODO: Remove this diagnostics when __SYCL_WARN_IMAGE_ASPECT is removed.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgcc-compat"
#endif // defined(__clang__)

  /// Indicates if all of the SYCL devices on this platform have the
  /// given feature.
  ///
  /// \param Aspect is one of the values in Table 4.20 of the SYCL 2020
  /// Provisional Spec.
  ///
  /// \return true if all of the SYCL devices on this platform have the
  /// given feature.
  bool has(aspect Aspect) const __SYCL_WARN_IMAGE_ASPECT(Aspect);

// TODO: Remove this diagnostics when __SYCL_WARN_IMAGE_ASPECT is removed.
#if defined(__clang__)
#pragma clang diagnostic pop
#endif // defined(__clang__)

  /// Return this platform's default context
  ///
  /// \return the default context
  context ext_oneapi_get_default_context() const;

  std::vector<device> ext_oneapi_get_composite_devices() const;

private:
  pi_native_handle getNative() const;

  std::shared_ptr<detail::platform_impl> impl;
  platform(std::shared_ptr<detail::platform_impl> impl) : impl(impl) {}

  platform(const device &Device);

  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);
  template <class Obj>
  friend const decltype(Obj::impl) &
  detail::getSyclObjImpl(const Obj &SyclObject);

  template <backend BackendName, class SyclObjectT>
  friend auto get_native(const SyclObjectT &Obj)
      -> backend_return_t<BackendName, SyclObjectT>;

  template <typename Param>
  typename detail::ABINeutralT_t<
      typename detail::is_platform_info_desc<Param>::return_type>
  get_info_impl() const;

  bool has_extension(detail::string_view ExtensionName) const;
}; // class platform
} // namespace _V1
} // namespace sycl

namespace std {
template <> struct hash<sycl::platform> {
  size_t operator()(const sycl::platform &p) const {
    return hash<std::shared_ptr<sycl::detail::platform_impl>>()(
        sycl::detail::getSyclObjImpl(p));
  }
};
} // namespace std
