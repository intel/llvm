//==------------------- device.hpp - SYCL device ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/aspects.hpp>
#include <sycl/backend_types.hpp>
#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/detail/info_desc_helpers.hpp>
#include <sycl/detail/owner_less_base.hpp>
#include <sycl/detail/pi.h>
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
#include <sycl/detail/string.hpp>
#include <sycl/detail/string_view.hpp>
#endif
#include <sycl/detail/util.hpp>
#include <sycl/device_selector.hpp>
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#include <sycl/info/info_desc.hpp>
#include <sycl/kernel_bundle_enums.hpp>
#include <sycl/platform.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <variant>
#include <vector>

namespace sycl {
inline namespace _V1 {
// Forward declarations
class device_selector;
template <backend BackendName, class SyclObjectT>
auto get_native(const SyclObjectT &Obj)
    -> backend_return_t<BackendName, SyclObjectT>;
namespace detail {
class device_impl;
auto getDeviceComparisonLambda();
} // namespace detail

enum class aspect;

namespace ext::oneapi {
// Forward declaration
class filter_selector;

enum class peer_access {
  access_supported = 0x0,
  atomics_supported = 0x1,
};

} // namespace ext::oneapi

/// The SYCL device class encapsulates a single SYCL device on which kernels
/// may be executed.
///
/// \ingroup sycl_api
class __SYCL_EXPORT device : public detail::OwnerLessBase<device> {
public:
  /// Constructs a SYCL device instance using the default device.
  device();

  /// Constructs a SYCL device instance from an OpenCL cl_device_id
  /// in accordance with the requirements described in 4.3.1.
  ///
  /// \param DeviceId is OpenCL device represented with cl_device_id
#ifdef __SYCL_INTERNAL_API
  explicit device(cl_device_id DeviceId);
#endif

  /// Constructs a SYCL device instance using the device selected
  /// by the DeviceSelector provided.
  ///
  /// \param DeviceSelector SYCL 1.2.1 device_selector to be used (see 4.6.1.1).
  __SYCL2020_DEPRECATED("SYCL 1.2.1 device selectors are deprecated. Please "
                        "use SYCL 2020 device selectors instead.")
  explicit device(const device_selector &DeviceSelector);

  /// Constructs a SYCL device instance using the device
  /// identified by the device selector provided.
  /// \param DeviceSelector is SYCL 2020 Device Selector, a simple callable that
  /// takes a device and returns an int
  template <typename DeviceSelector,
            typename =
                detail::EnableIfSYCL2020DeviceSelectorInvocable<DeviceSelector>>
  explicit device(const DeviceSelector &deviceSelector)
      : device(detail::select_device(deviceSelector)) {}

  bool operator==(const device &rhs) const { return impl == rhs.impl; }

  bool operator!=(const device &rhs) const { return !(*this == rhs); }

  device(const device &rhs) = default;

  device(device &&rhs) = default;

  device &operator=(const device &rhs) = default;

  device &operator=(device &&rhs) = default;

  void ext_oneapi_enable_peer_access(const device &peer);
  void ext_oneapi_disable_peer_access(const device &peer);
  bool
  ext_oneapi_can_access_peer(const device &peer,
                             ext::oneapi::peer_access value =
                                 ext::oneapi::peer_access::access_supported);

  /// Get instance of device
  ///
  /// \return a valid cl_device_id instance in accordance with the requirements
  /// described in 4.3.1.
#ifdef __SYCL_INTERNAL_API
  cl_device_id get() const;
#endif

  /// Check if device is a host device
  ///
  /// \return true if SYCL device is a host device
  __SYCL2020_DEPRECATED(
      "is_host() is deprecated as the host device is no longer supported.")
  bool is_host() const;

  /// Check if device is a CPU device
  ///
  /// \return true if SYCL device is a CPU device
  bool is_cpu() const;

  /// Check if device is a GPU device
  ///
  /// \return true if SYCL device is a GPU device
  bool is_gpu() const;

  /// Check if device is an accelerator device
  ///
  /// \return true if SYCL device is an accelerator device
  bool is_accelerator() const;

  /// Get associated SYCL platform
  ///
  /// If this SYCL device is an OpenCL device then the SYCL platform
  /// must encapsulate the OpenCL cl_plaform_id associated with the
  /// underlying OpenCL cl_device_id of this SYCL device. If this SYCL device
  /// is a host device then the SYCL platform must be a host platform.
  /// The value returned must be equal to that returned by
  /// get_info<info::device::platform>().
  ///
  /// \return The associated SYCL platform.
  platform get_platform() const;

  /// Partition device into sub devices
  ///
  /// Available only when prop is info::partition_property::partition_equally.
  /// If this SYCL device does not support
  /// info::partition_property::partition_equally a feature_not_supported
  /// exception must be thrown.
  ///
  /// \param ComputeUnits is a desired count of compute units in each sub
  /// device.
  /// \return A vector class of sub devices partitioned from this SYCL
  /// device equally based on the ComputeUnits parameter.
  template <info::partition_property prop>
  std::vector<device> create_sub_devices(size_t ComputeUnits) const;

  /// Partition device into sub devices
  ///
  /// Available only when prop is info::partition_property::partition_by_counts.
  /// If this SYCL device does not support
  /// info::partition_property::partition_by_counts a feature_not_supported
  /// exception must be thrown.
  ///
  /// \param Counts is a std::vector of desired compute units in sub devices.
  /// \return a std::vector of sub devices partitioned from this SYCL device by
  /// count sizes based on the Counts parameter.
  template <info::partition_property prop>
  std::vector<device>
  create_sub_devices(const std::vector<size_t> &Counts) const;

  /// Partition device into sub devices
  ///
  /// Available only when prop is
  /// info::partition_property::partition_by_affinity_domain. If this SYCL
  /// device does not support
  /// info::partition_property::partition_by_affinity_domain or the SYCL device
  /// does not support info::affinity_domain provided a feature_not_supported
  /// exception must be thrown.
  ///
  /// \param AffinityDomain is one of the values described in Table 4.20 of SYCL
  /// Spec
  /// \return a vector class of sub devices partitioned from this SYCL
  /// device by affinity domain based on the AffinityDomain parameter
  template <info::partition_property prop>
  std::vector<device>
  create_sub_devices(info::partition_affinity_domain AffinityDomain) const;

  /// Partition device into sub devices
  ///
  /// Available only when prop is
  /// info::partition_property::ext_intel_partition_by_cslice. If this SYCL
  /// device does not support
  /// info::partition_property::ext_intel_partition_by_cslice a
  /// feature_not_supported exception must be thrown.
  ///
  /// \return a vector class of sub devices partitioned from this SYCL
  /// device at a granularity of "cslice" (compute slice).
  template <info::partition_property prop>
  std::vector<device> create_sub_devices() const;

  /// Queries this SYCL device for information requested by the template
  /// parameter param
  ///
  /// Specializations of info::param_traits must be defined in accordance with
  /// the info parameters in Table 4.20 of SYCL Spec to facilitate returning the
  /// type associated with the param parameter.
  ///
  /// \return device info of type described in Table 4.20.
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  template <typename Param>
  typename detail::is_device_info_desc<Param>::return_type get_info() const {
    return detail::convert_from_abi_neutral(get_info_impl<Param>());
  }
#else
  template <typename Param>
  detail::ABINeutralT_t<
      typename detail::is_device_info_desc<Param>::return_type>
  get_info() const;
#endif

  /// Check SYCL extension support by device
  ///
  /// \param extension_name is a name of queried extension.
  /// \return true if SYCL device supports the extension.
  __SYCL2020_DEPRECATED("use device::has() function with aspects APIs instead")
  bool has_extension(const std::string &extension_name) const;

  /// Query available SYCL devices
  ///
  /// The returned std::vector must contain a single SYCL device
  /// that is a host device, permitted by the deviceType parameter
  ///
  /// \param deviceType is one of the values described in A.3 of SYCL Spec
  /// \return a std::vector containing all SYCL devices available in the system
  /// of the device type specified
  static std::vector<device>
  get_devices(info::device_type deviceType = info::device_type::all);

  /// Returns the backend associated with this device.
  ///
  /// \return the backend associated with this device.
  backend get_backend() const noexcept;

// Clang may warn about the use of diagnose_if in __SYCL_WARN_IMAGE_ASPECT, so
// we disable that warning as we make appropriate checks to ensure its
// existence.
// TODO: Remove this diagnostics when __SYCL_WARN_IMAGE_ASPECT is removed.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgcc-compat"
#endif // defined(__clang__)

  /// Indicates if the SYCL device has the given feature.
  ///
  /// \param Aspect is one of the values in Table 4.20 of the SYCL 2020
  /// Provisional Spec.
  ///
  /// \return true if the SYCL device has the given feature.
  bool has(aspect Aspect) const __SYCL_WARN_IMAGE_ASPECT(Aspect);

  /// Indicates if the SYCL device architecture equals to the one passed to
  /// the function.
  ///
  /// \param arch is one of the architectures from architecture enum described
  /// in sycl_ext_oneapi_device_architecture specification.
  ///
  /// \return true if the SYCL device architecture equals to the one passed to
  /// the function.
  bool ext_oneapi_architecture_is(ext::oneapi::experimental::architecture arch);

  /// Indicates if the SYCL device architecture is in the category passed
  /// to the function.
  ///
  /// \param category is one of the architecture categories from arch_category
  /// enum described in sycl_ext_oneapi_device_architecture specification.
  ///
  /// \return true if the SYCL device architecture is in the category passed to
  /// the function.
  bool
  ext_oneapi_architecture_is(ext::oneapi::experimental::arch_category category);

  /// kernel_compiler extension

  /// Indicates if the device can compile a kernel for the given language.
  ///
  /// \param Language is one of the values from the
  /// kernel_bundle::source_language enumeration described in the
  /// sycl_ext_oneapi_kernel_compiler specification
  ///
  /// \return true only if the device supports kernel bundles written in the
  /// source language `lang`.
  bool
  ext_oneapi_can_compile(ext::oneapi::experimental::source_language Language);

  /// Indicates if the device supports a given feature when compiling the OpenCL
  /// C language
  ///
  /// \param Feature
  ///
  /// \return true if supported
  bool ext_oneapi_supports_cl_c_feature(const std::string &Feature);

  /// Indicates if the device supports kernel bundles written in a particular
  /// OpenCL C version
  ///
  /// \param Version
  ///
  /// \return true only if the device supports kernel bundles written in the
  /// version identified by `Version`.
  bool ext_oneapi_supports_cl_c_version(
      const ext::oneapi::experimental::cl_version &Version) const;

  /// If the device supports kernel bundles using the OpenCL extension
  /// identified by `name` and if `version` is not a null pointer, the supported
  /// version of the extension is written to `version`.
  ///
  /// \return true only if the device supports kernel bundles using the OpenCL
  /// extension identified by `name`.
  bool ext_oneapi_supports_cl_extension(
      const std::string &name,
      ext::oneapi::experimental::cl_version *version = nullptr) const;

  /// Retrieve the OpenCl Device Profile
  ///
  /// \return If the device supports kernel bundles written in
  /// `source_language::opencl`, returns the name of the OpenCL profile that is
  /// supported. The profile name is the same string that is returned by the
  /// query `CL_DEVICE_PROFILE`, as defined in section 4.2 "Querying Devices" of
  /// the OpenCL specification. If the device does not support kernel bundles
  /// written in `source_language::opencl`, returns the empty string.
  std::string ext_oneapi_cl_profile() const;

// TODO: Remove this diagnostics when __SYCL_WARN_IMAGE_ASPECT is removed.
#if defined(__clang__)
#pragma clang diagnostic pop
#endif // defined(__clang__)

private:
  std::shared_ptr<detail::device_impl> impl;
  device(std::shared_ptr<detail::device_impl> impl) : impl(impl) {}

  pi_native_handle getNative() const;

  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend typename std::add_pointer_t<typename decltype(T::impl)::element_type>
  detail::getRawSyclObjImpl(const T &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);

  template <backend BackendName, class SyclObjectT>
  friend auto get_native(const SyclObjectT &Obj)
      -> backend_return_t<BackendName, SyclObjectT>;

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  template <typename Param>
  typename detail::ABINeutralT_t<
      typename detail::is_device_info_desc<Param>::return_type>
  get_info_impl() const;
#endif
};

} // namespace _V1
} // namespace sycl

namespace std {
template <> struct hash<sycl::device> {
  size_t operator()(const sycl::device &Device) const {
    return hash<std::shared_ptr<sycl::detail::device_impl>>()(
        sycl::detail::getSyclObjImpl(Device));
  }
};
} // namespace std
