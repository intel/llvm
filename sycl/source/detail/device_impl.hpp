//==----------------- device_impl.hpp - SYCL device ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/aspects.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/kernel_bundle.hpp>
#include <CL/sycl/stl.hpp>
#include <detail/device_info.hpp>
#include <detail/platform_impl.hpp>

#include <memory>
#include <mutex>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

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
  explicit device_impl(pi_native_handle, const plugin &Plugin);

  /// Constructs a SYCL device instance using the provided
  /// PI device instance.
  explicit device_impl(RT::PiDevice Device, PlatformImplPtr Platform);

  /// Constructs a SYCL device instance using the provided
  /// PI device instance.
  explicit device_impl(RT::PiDevice Device, const plugin &Plugin);

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
  RT::PiDevice &getHandleRef() {
    if (MIsHostDevice)
      throw invalid_object_error("This instance of device is a host instance",
                                 PI_INVALID_DEVICE);

    return MDevice;
  }

  /// Get constant reference to PI device
  ///
  /// For host device an exception is thrown
  ///
  /// \return constant reference to PI device
  const RT::PiDevice &getHandleRef() const {
    if (MIsHostDevice)
      throw invalid_object_error("This instance of device is a host instance",
                                 PI_INVALID_DEVICE);

    return MDevice;
  }

  /// Check if SYCL device is a host device
  ///
  /// \return true if SYCL device is a host device
  bool is_host() const { return MIsHostDevice; }

  /// Check if device is a CPU device
  ///
  /// \return true if SYCL device is a CPU device
  bool is_cpu() const { return (!is_host() && (MType == PI_DEVICE_TYPE_CPU)); }

  /// Check if device is a GPU device
  ///
  /// \return true if SYCL device is a GPU device
  bool is_gpu() const { return (!is_host() && (MType == PI_DEVICE_TYPE_GPU)); }

  /// Check if device is an accelerator device
  ///
  /// \return true if SYCL device is an accelerator device
  bool is_accelerator() const {
    return (!is_host() && (MType == PI_DEVICE_TYPE_ACC));
  }

  /// Return device type
  ///
  /// \return the type of the device
  RT::PiDeviceType get_device_type() const { return MType; }

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
  const plugin &getPlugin() const { return MPlatform->getPlugin(); }

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
  template <info::device param>
  typename info::param_traits<info::device, param>::return_type
  get_info() const {
    if (is_host()) {
      return get_device_info_host<param>();
    }
    return get_device_info<
        typename info::param_traits<info::device, param>::return_type,
        param>::get(this->getHandleRef(), this->getPlugin());
  }

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

  /// Gets the single instance of the Host Device
  ///
  /// \return the host device_impl singleton
  static std::shared_ptr<device_impl> getHostDeviceImpl();

  bool isAssertFailSupported() const;

  bool isRootDevice() const { return MRootDevice == nullptr; }
  
  std::string getDeviceName() const;

private:
  explicit device_impl(pi_native_handle InteropDevice, RT::PiDevice Device,
                       PlatformImplPtr Platform, const plugin &Plugin);
  RT::PiDevice MDevice = 0;
  RT::PiDeviceType MType;
  RT::PiDevice MRootDevice = nullptr;
  bool MIsHostDevice;
  PlatformImplPtr MPlatform;
  bool MIsAssertFailSupported = false;
  mutable std::string MDeviceName;
  mutable std::once_flag MDeviceNameFlag;
}; // class device_impl

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
