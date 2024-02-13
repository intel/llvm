//==------------------- device.hpp - SYCL device ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/aspects.hpp>                   // for aspect
#include <sycl/backend_types.hpp>             // for backend
#include <sycl/detail/defines_elementary.hpp> // for __SY...
#include <sycl/detail/export.hpp>             // for __SY...
#include <sycl/detail/info_desc_helpers.hpp>  // for is_d...
#include <sycl/detail/owner_less_base.hpp>    // for Owne...
#include <sycl/detail/pi.h>                   // for pi_n...
#include <sycl/detail/string.hpp>             // for c++11 abi compatibility
#include <sycl/detail/string_view.hpp>        // for c++11 abi compatibility
#include <sycl/device_selector.hpp>           // for Enab...
#include <sycl/ext/oneapi/experimental/device_architecture.hpp> // for arch...
#include <sycl/info/info_desc.hpp>                              // for part...
#include <sycl/platform.hpp>                                    // for plat...

#include <cstddef>     // for size_t
#include <memory>      // for shar...
#include <string>      // for string
#include <type_traits> // for add_...
#include <typeinfo>
#include <variant> // for hash
#include <vector>  // for vector

namespace std {
// We need special handling of std::string to handle ABI incompatibility
// for get_info<>() when it returns std::string and vector<std::string>.
// For this purpose, get_info_internal<>() is created to handle special
// cases, and it is only called internally and not exposed to the user.
// The following ReturnType structure is intended for general return type,
// and special return types (std::string and vector of it).
template <typename T>
struct ReturnType {
  using type = T;
};

template <>
struct ReturnType<std::string> {
  using type = sycl::_V1::detail::string;
};

template <>
struct ReturnType<std::vector<std::string>> {
  using type = std::vector<sycl::_V1::detail::string>;
};
} // namespace std

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
    // For C++11_ABI compatibility, we handle these string Param types
    // separately.
    if constexpr (std::is_same_v<std::string,
                                 typename detail::is_device_info_desc<
                                     Param>::return_type>) {
      // auto *Map = getMap();
      // auto iter = Map->find(typeid(Param).name());
      // if (iter == Map->end()) {
      //   throw sycl::invalid_parameter_error("unsupported device info requested",
      //                                       PI_ERROR_INVALID_OPERATION);
      // }
      // DeviceProperty PropertyName = iter->second;
      detail::string Info = get_device_info<Param>();
      return Info.c_str();
    } else if constexpr (std::is_same_v<std::vector<std::string>,
                                        typename detail::is_device_info_desc<
                                            Param>::return_type>) {
      // return value is std::vector<std::string>
      // auto *Map = getMap();
      // auto iter = Map->find(typeid(Param).name());
      // if (iter == Map->end()) {
      //   throw sycl::invalid_parameter_error("unsupported device info requested",
      //                                       PI_ERROR_INVALID_OPERATION);
      // }
      // DeviceProperty PropertyName = iter->second;
      std::vector<detail::string> Info = get_device_info_vector<Param>();
      std::vector<std::string> Res;
      Res.reserve(Info.size());
      for (detail::string &Str : Info) {
        Res.push_back(Str.c_str());
      }
      return Res;
    } else
    return get_info_impl<Param>();
  }

#else
  template <typename Param>
  typename detail::is_device_info_desc<Param>::return_type get_info() const;
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
  typename std::ReturnType<typename detail::is_device_info_desc<Param>::return_type>::type
  get_info_impl() const;

  // proxy of get_info_internal() to handle C++11-ABI compatibility separately.
  template <typename Param>
  detail::string get_device_info() const;
  
  template <typename Param>
  std::vector<detail::string>
  get_device_info_vector() const;

  // static const std::unordered_map<const char *, DeviceProperty> *getMap() {
  //   static const auto *map =
  //       new std::unordered_map<const char *, DeviceProperty>{
  //           {typeid(info::device::backend_version).name(),
  //            DeviceProperty::BACKEND_VERSION},
  //           {typeid(info::device::backend_version).name(),
  //            DeviceProperty::BACKEND_VERSION},
  //           {typeid(info::device::built_in_kernels).name(),
  //            DeviceProperty::BUILT_IN_KERNELS},
  //           {typeid(info::device::driver_version).name(),
  //            DeviceProperty::DRIVER_VERSION},
  //           {typeid(info::device::extensions).name(),
  //            DeviceProperty::EXTENSIONS},
  //           {typeid(info::device::ext_intel_pci_address).name(),
  //            DeviceProperty::EXT_INTEL_PCI_ADDRESS},
  //           {typeid(info::device::name).name(), DeviceProperty::NAME},
  //           {typeid(info::device::opencl_c_version).name(),
  //            DeviceProperty::OPENCL_C_VERSION},
  //           {typeid(info::device::profile).name(), DeviceProperty::PROFILE},
  //           {typeid(info::device::vendor).name(), DeviceProperty::VENDOR},
  //           { typeid(info::device::version).name(),
  //             DeviceProperty::VERSION }};
  //   return map;
  // }
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
