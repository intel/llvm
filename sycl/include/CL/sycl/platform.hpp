//==---------------- platform.hpp - SYCL platform --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/aspects.hpp>
#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/backend_traits.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/stl.hpp>

// 4.6.2 Platform class
#include <utility>
__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
// TODO: make code thread-safe

// Forward declaration
class device_selector;
class device;
namespace detail {
class platform_impl;
}
template <backend Backend, class SyclT>
auto get_native(const SyclT &Obj) -> backend_return_t<Backend, SyclT>;

/// Encapsulates a SYCL platform on which kernels may be executed.
///
/// \ingroup sycl_api
class __SYCL_EXPORT platform {
public:
  /// Constructs a SYCL platform as a host platform.
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

  /// Constructs a SYCL platform instance using device selector.
  ///
  /// One of the SYCL devices that is associated with the constructed SYCL
  /// platform instance must be the SYCL device that is produced from the
  /// provided device selector.
  ///
  /// \param DeviceSelector is an instance of SYCL device_selector.
  explicit platform(const device_selector &DeviceSelector);

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
  bool has_extension(const std::string &ExtensionName) const;

  /// Checks if this SYCL platform is a host platform.
  ///
  /// \return true if this SYCL platform is a host platform.
  bool is_host() const;

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
  template <info::platform param>
  typename info::param_traits<info::platform, param>::return_type
  get_info() const;

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

  /// Gets the native handle of the SYCL platform.
  ///
  /// \return a native handle, the type of which defined by the backend.
  template <backend Backend>
  __SYCL_DEPRECATED("Use SYCL 2020 sycl::get_native free function")
  backend_return_t<Backend, platform> get_native() const {
    return sycl::get_native<Backend, platform>(*this);
  }

  /// Indicates if all of the SYCL devices on this platform have the
  /// given feature.
  ///
  /// \param Aspect is one of the values in Table 4.20 of the SYCL 2020
  /// Provisional Spec.
  ///
  /// \return true if all of the SYCL devices on this platform have the
  /// given feature.
  bool has(aspect Aspect) const;

  /// Return this platform's default context
  ///
  /// \return the default context
  context ext_oneapi_get_default_context() const;

private:
  pi_native_handle getNative() const;

  std::shared_ptr<detail::platform_impl> impl;
  platform(std::shared_ptr<detail::platform_impl> impl) : impl(impl) {}

  template <backend Backend, class SyclT>
  friend auto get_native(const SyclT &Obj) -> backend_return_t<Backend, SyclT>;

  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);
  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);

}; // class platform
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

namespace std {
template <> struct hash<cl::sycl::platform> {
  size_t operator()(const cl::sycl::platform &p) const {
    return hash<std::shared_ptr<cl::sycl::detail::platform_impl>>()(
        cl::sycl::detail::getSyclObjImpl(p));
  }
};
} // namespace std
