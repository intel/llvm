//==----------------- device_impl.hpp - SYCL device ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/platform_impl.hpp>
#include <sycl/aspects.hpp>
#include <sycl/detail/cl.h>
#include <sycl/detail/pi.hpp>
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#include <sycl/ext/oneapi/experimental/forward_progress.hpp>
#include <sycl/kernel_bundle.hpp>

#include <memory>
#include <mutex>
#include <utility>

namespace sycl {
inline namespace _V1 {

// Forward declaration
class platform;

namespace detail {

// Forward declaration
class platform_impl;
using PlatformImplPtr = std::shared_ptr<platform_impl>;

// TODO: Make code thread-safe
class device_impl {
public:
  /// Constructs a SYCL device instance as a host device.
  device_impl();

  /// Constructs a SYCL device instance using the provided raw device handle.
  explicit device_impl(pi_native_handle, const PluginPtr &Plugin);

  /// Constructs a SYCL device instance using the provided
  /// PI device instance.
  explicit device_impl(sycl::detail::pi::PiDevice Device,
                       PlatformImplPtr Platform);

  /// Constructs a SYCL device instance using the provided
  /// PI device instance.
  explicit device_impl(sycl::detail::pi::PiDevice Device,
                       const PluginPtr &Plugin);

  ~device_impl();

  /// Get instance of OpenCL device
  ///
  /// \return a valid cl_device_id instance in accordance with the
  /// requirements described in 4.3.1.
  cl_device_id get() const;

  /// Get reference to PI device
  ///
  /// For host device an exception is thrown
  ///
  /// \return non-constant reference to PI device
  sycl::detail::pi::PiDevice &getHandleRef() { return MDevice; }

  /// Get constant reference to PI device
  ///
  /// For host device an exception is thrown
  ///
  /// \return constant reference to PI device
  const sycl::detail::pi::PiDevice &getHandleRef() const { return MDevice; }

  /// Check if device is a CPU device
  ///
  /// \return true if SYCL device is a CPU device
  bool is_cpu() const { return MType == PI_DEVICE_TYPE_CPU; }

  /// Check if device is a GPU device
  ///
  /// \return true if SYCL device is a GPU device
  bool is_gpu() const { return MType == PI_DEVICE_TYPE_GPU; }

  /// Check if device is an accelerator device
  ///
  /// \return true if SYCL device is an accelerator device
  bool is_accelerator() const { return MType == PI_DEVICE_TYPE_ACC; }

  /// Return device type
  ///
  /// \return the type of the device
  sycl::detail::pi::PiDeviceType get_device_type() const { return MType; }

  /// Get associated SYCL platform
  ///
  /// If this SYCL device is an OpenCL device then the SYCL platform
  /// must encapsulate the OpenCL cl_plaform_id associated with the
  /// underlying OpenCL cl_device_id of this SYCL device. If this SYCL device
  /// is a host device then the SYCL platform must be a host platform.
  /// The value returned must be equal to that returned
  /// by get_info<info::device::platform>().
  ///
  /// \return The associated SYCL platform.
  platform get_platform() const;

  /// \return the associated plugin with this device.
  const PluginPtr &getPlugin() const { return MPlatform->getPlugin(); }

  /// Check SYCL extension support by device
  ///
  /// \param ExtensionName is a name of queried extension.
  /// \return true if SYCL device supports the extension.
  bool has_extension(const std::string &ExtensionName) const;

  std::vector<device>
  create_sub_devices(const cl_device_partition_property *Properties,
                     size_t SubDevicesCount) const;

  /// Partition device into sub devices
  ///
  /// If this SYCL device does not support
  /// info::partition_property::partition_equally a feature_not_supported
  /// exception must be thrown.
  ///
  /// \param ComputeUnits is a desired count of compute units in each sub
  /// device.
  /// \return A vector class of sub devices partitioned equally from this
  /// SYCL device based on the ComputeUnits parameter.
  std::vector<device> create_sub_devices(size_t ComputeUnits) const;

  /// Partition device into sub devices
  ///
  /// If this SYCL device does not support
  /// info::partition_property::partition_by_counts a feature_not_supported
  /// exception must be thrown.
  ///
  /// \param Counts is a std::vector of desired compute units in sub devices.
  /// \return a std::vector of sub devices partitioned from this SYCL device
  /// by count sizes based on the Counts parameter.
  std::vector<device>
  create_sub_devices(const std::vector<size_t> &Counts) const;

  /// Partition device into sub devices
  ///
  /// If this SYCL device does not support
  /// info::partition_property::partition_by_affinity_domain or the SYCL
  /// device does not support info::affinity_domain provided a
  /// feature_not_supported exception must be thrown.
  ///
  /// \param AffinityDomain is one of the values described in Table 4.20 of
  /// SYCL Spec
  /// \return a vector class of sub devices partitioned from this SYCL device
  /// by affinity domain based on the AffinityDomain parameter
  std::vector<device>
  create_sub_devices(info::partition_affinity_domain AffinityDomain) const;

  /// Partition device into sub devices
  ///
  /// If this SYCL device does not support
  /// info::partition_property::ext_intel_partition_by_cslice a
  /// feature_not_supported exception must be thrown.
  ///
  /// \return a vector class of sub devices partitioned from this SYCL
  /// device at a granularity of "cslice" (compute slice).
  std::vector<device> create_sub_devices() const;

  /// Check if desired partition property supported by device
  ///
  /// \param Prop is one of info::partition_property::(partition_equally,
  /// partition_by_counts, partition_by_affinity_domain)
  /// \return true if Prop is supported by device.
  bool is_partition_supported(info::partition_property Prop) const;

  /// Queries this SYCL device for information requested by the template
  /// parameter param
  ///
  /// Specializations of info::param_traits must be defined in accordance
  /// with the info parameters in Table 4.20 of SYCL Spec to facilitate
  /// returning the type associated with the param parameter.
  ///
  /// \return device info of type described in Table 4.20.
  template <typename Param> typename Param::return_type get_info() const;

  /// Queries SYCL queue for SYCL backend-specific information.
  ///
  /// The return type depends on information being queried.
  template <typename Param>
  typename Param::return_type get_backend_info() const;

  /// Check if affinity partitioning by specified domain is supported by
  /// device
  ///
  /// \param AffinityDomain is one of the values described in Table 4.20 of
  /// SYCL Spec
  /// \return true if AffinityDomain is supported by device.
  bool
  is_affinity_supported(info::partition_affinity_domain AffinityDomain) const;

  /// Gets the native handle of the SYCL device.
  ///
  /// \return a native handle.
  pi_native_handle getNative() const;

  /// Indicates if the SYCL device has the given feature.
  ///
  /// \param Aspect is one of the values in Table 4.20 of the SYCL 2020
  /// Provisional Spec.
  //
  /// \return true if the SYCL device has the given feature.
  bool has(aspect Aspect) const;

  bool isAssertFailSupported() const;

  bool isRootDevice() const { return MRootDevice == nullptr; }

  std::string getDeviceName() const;

  bool
  extOneapiArchitectureIs(ext::oneapi::experimental::architecture Arch) const {
    return Arch == getDeviceArch();
  }

  bool extOneapiArchitectureIs(
      ext::oneapi::experimental::arch_category Category) const {
    std::optional<ext::oneapi::experimental::architecture> CategoryMinArch =
        get_category_min_architecture(Category);
    std::optional<ext::oneapi::experimental::architecture> CategoryMaxArch =
        get_category_max_architecture(Category);
    if (CategoryMinArch.has_value() && CategoryMaxArch.has_value())
      return CategoryMinArch <= getDeviceArch() &&
             getDeviceArch() <= CategoryMaxArch;
    return false;
  }

  bool extOneapiCanCompile(ext::oneapi::experimental::source_language Language);

  // Returns all guarantees that are either equal to guarantee or weaker than
  // it. E.g if guarantee == parallel, it returns the vector {weakly_parallel,
  // parallel}.
  template <typename ReturnT>
  static ReturnT getProgressGuaranteesUpTo(
      ext::oneapi::experimental::forward_progress_guarantee guarantee) {
    const int forwardProgressGuaranteeSize = 3;
    int guaranteeVal = static_cast<int>(guarantee);
    ReturnT res;
    res.reserve(forwardProgressGuaranteeSize - guaranteeVal);
    for (int currentGuarantee = forwardProgressGuaranteeSize - 1;
         currentGuarantee >= guaranteeVal; --currentGuarantee) {
      res.emplace_back(
          static_cast<ext::oneapi::experimental::forward_progress_guarantee>(
              currentGuarantee));
    }
    return res;
  }

  static sycl::ext::oneapi::experimental::forward_progress_guarantee
  getHostProgressGuarantee(
      sycl::ext::oneapi::experimental::execution_scope threadScope,
      sycl::ext::oneapi::experimental::execution_scope coordinationScope);

  sycl::ext::oneapi::experimental::forward_progress_guarantee
  getProgressGuarantee(
      ext::oneapi::experimental::execution_scope threadScope,
      ext::oneapi::experimental::execution_scope coordinationScope) const;

  bool supportsForwardProgress(
      ext::oneapi::experimental::forward_progress_guarantee guarantee,
      ext::oneapi::experimental::execution_scope threadScope,
      ext::oneapi::experimental::execution_scope coordinationScope) const;

  ext::oneapi::experimental::forward_progress_guarantee
  getImmediateProgressGuarantee(
      ext::oneapi::experimental::execution_scope coordination_scope) const;

  /// Gets the current device timestamp
  /// @throw sycl::feature_not_supported if feature is not supported on device
  uint64_t getCurrentDeviceTime();

  /// Check clGetDeviceAndHostTimer is available for fallback profiling

  bool isGetDeviceAndHostTimerSupported();

  /// Get the backend of this device
  backend getBackend() const { return MPlatform->getBackend(); }

  /// @brief  Get the platform impl serving this device
  /// @return PlatformImplPtr
  PlatformImplPtr getPlatformImpl() const { return MPlatform; }

  /// Get device info string
  std::string
  get_device_info_string(sycl::detail::pi::PiDeviceInfo InfoCode) const;

  /// Get device architecture
  ext::oneapi::experimental::architecture getDeviceArch() const;

private:
  explicit device_impl(pi_native_handle InteropDevice,
                       sycl::detail::pi::PiDevice Device,
                       PlatformImplPtr Platform, const PluginPtr &Plugin);
  sycl::detail::pi::PiDevice MDevice = 0;
  sycl::detail::pi::PiDeviceType MType;
  sycl::detail::pi::PiDevice MRootDevice = nullptr;
  PlatformImplPtr MPlatform;
  bool MIsAssertFailSupported = false;
  mutable std::string MDeviceName;
  mutable std::once_flag MDeviceNameFlag;
  mutable ext::oneapi::experimental::architecture MDeviceArch{};
  mutable std::once_flag MDeviceArchFlag;
  std::pair<uint64_t, uint64_t> MDeviceHostBaseTime{0, 0};
}; // class device_impl

} // namespace detail
} // namespace _V1
} // namespace sycl
