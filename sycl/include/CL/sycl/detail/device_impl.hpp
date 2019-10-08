//==----------------- device_impl.hpp - SYCL device ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/device_info.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/stl.hpp>
#include <memory>

namespace cl {
namespace sycl {

// Forward declaration
class platform;

namespace detail {

// Forward declaration
class platform_impl;
class platform_impl_pi;

// TODO: Make code thread-safe
class device_impl {
public:
  /// Constructs a SYCL device instance as a host device.
  device_impl();
  /// Constructs a SYCL device instance using the provided
  /// PI device instance.
  explicit device_impl(RT::PiDevice Device);

  ~device_impl();

  /// Get instance of OpenCL device
  ///
  /// @return a valid cl_device_id instance in accordance with the requirements
  /// described in 4.3.1.
  cl_device_id get() const;

  /// Get reference to PI device
  ///
  /// For host device an exception is thrown
  ///
  /// @return non-constant reference to PI device
  RT::PiDevice &getHandleRef() {
    if (MIsHostDevice)
      throw invalid_object_error("This instance of device is a host instance");

    return MDevice;
  }

  /// Get constant reference to PI device
  ///
  /// For host device an exception is thrown
  ///
  /// @return constant reference to PI device
  const RT::PiDevice &getHandleRef() const {
    if (MIsHostDevice)
      throw invalid_object_error("This instance of device is a host instance");

    return MDevice;
  }

  /// Check if SYCL device is a host device
  ///
  /// @return true if SYCL device is a host device
  bool is_host() const { return MIsHostDevice; }

  /// Check if device is a CPU device
  ///
  /// @return true if SYCL device is a CPU device
  bool is_cpu() const { return (MType == PI_DEVICE_TYPE_CPU); }

  /// Check if device is a GPU device
  ///
  /// @return true if SYCL device is a GPU device
  bool is_gpu() const { return (MType == PI_DEVICE_TYPE_GPU); }

  /// Check if device is an accelerator device
  ///
  /// @return true if SYCL device is an accelerator device
  bool is_accelerator() const { return (MType == PI_DEVICE_TYPE_ACC); }

  /// Get associated SYCL platform
  ///
  /// If this SYCL device is an OpenCL device then the SYCL platform
  /// must encapsulate the OpenCL cl_plaform_id associated with the
  /// underlying OpenCL cl_device_id of this SYCL device. If this SYCL device
  /// is a host device then the SYCL platform must be a host platform.
  /// The value returned must be equal to that returned
  /// by get_info<info::device::platform>().
  ///
  /// @return The associated SYCL platform.
  platform get_platform() const;

  /// Check SYCL extension support by device
  ///
  /// @param ExtensionName is a name of queried extension.
  /// @return true if SYCL device supports the extension.
  bool has_extension(const string_class &ExtensionName) const;

  vector_class<device>
  create_sub_devices(const cl_device_partition_property *Properties,
                     size_t SubDevicesCount) const;

  /// Partition device into sub devices
  ///
  /// If this SYCL device does not support info::partition_property::partition_equally
  /// a feature_not_supported exception must be thrown.
  ///
  /// @param ComputeUnits is a desired count of compute units in each sub device.
  /// @return A vector class of sub devices partitioned equally from this
  /// SYCL device based on the ComputeUnits parameter.
  vector_class<device> create_sub_devices(size_t ComputeUnits) const;

  /// Partition device into sub devices
  ///
  /// If this SYCL device does not support info::partition_property::partition_by_counts
  /// a feature_not_supported exception must be thrown.
  ///
  /// @param Counts is a vector_class of desired compute units in sub devices.
  /// @return a vector_class of sub devices partitioned from this SYCL device
  /// by count sizes based on the Counts parameter.
  vector_class<device>
  create_sub_devices(const vector_class<size_t> &Counts) const;

  /// Partition device into sub devices
  ///
  /// If this SYCL device does not support info::partition_property::partition_by_affinity_domain
  /// or the SYCL device does not support info::affinity_domain provided
  /// a feature_not_supported exception must be thrown.
  ///
  /// @param AffinityDomain is one of the values described in Table 4.20 of SYCL Spec
  /// @return a vector class of sub devices partitioned from this SYCL device
  /// by affinity domain based on the AffinityDomain parameter
  vector_class<device>
  create_sub_devices(info::partition_affinity_domain AffinityDomain) const;

  /// Check if desired partition property supported by device
  ///
  /// @param Prop is one of info::partition_property::(partition_equally,
  /// partition_by_counts, partition_by_affinity_domain)
  /// @return true if Prop is supported by device.
  bool is_partition_supported(info::partition_property Prop) const;

  /// Queries this SYCL device for information requested by the template parameter param
  ///
  /// Specializations of info::param_traits must be defined in accordance
  /// with the info parameters in Table 4.20 of SYCL Spec to facilitate
  /// returning the type associated with the param parameter.
  ///
  /// @return device info of type described in Table 4.20.
  template <info::device param>
  typename info::param_traits<info::device, param>::return_type
  get_info() const {
    if (is_host()) {
      return get_device_info_host<param>();
    }
    return get_device_info<
        typename info::param_traits<info::device, param>::return_type,
        param>::_(this->getHandleRef());
  }

  /// Check if affinity partitioning by specified domain is supported by device
  ///
  /// @param AffinityDomain is one of the values described in Table 4.20 of SYCL Spec
  /// @return true if AffinityDomain is supported by device.
  bool
  is_affinity_supported(info::partition_affinity_domain AffinityDomain) const;

private:
  RT::PiDevice MDevice = 0;
  RT::PiDeviceType MType;
  bool MIsRootDevice = false;
  bool MIsHostDevice;
}; // class device_impl

} // namespace detail
} // namespace sycl
} // namespace cl
