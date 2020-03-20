//==------------------- device.hpp - SYCL device ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/platform.hpp>
#include <CL/sycl/stl.hpp>
#include <memory>
#include <utility>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
// Forward declarations
class device_selector;
namespace detail {
class device_impl;
}
class device {
public:
  /// Constructs a SYCL device instance as a host device.
  device();

  /// Constructs a SYCL device instance from an OpenCL cl_device_id
  /// in accordance with the requirements described in 4.3.1.
  ///
  /// \param DeviceId is OpenCL device represented with cl_device_id
  explicit device(cl_device_id DeviceId);

  /// Constructs a SYCL device instance using the device selected
  /// by the DeviceSelector provided.
  ///
  /// \param DeviceSelector SYCL device selector to be used (see 4.6.1.1).
  explicit device(const device_selector &DeviceSelector);

  bool operator==(const device &rhs) const { return impl == rhs.impl; }

  bool operator!=(const device &rhs) const { return !(*this == rhs); }

  device(const device &rhs) = default;

  device(device &&rhs) = default;

  device &operator=(const device &rhs) = default;

  device &operator=(device &&rhs) = default;

  /// Get instance of device
  ///
  /// \return a valid cl_device_id instance in accordance with the requirements
  /// described in 4.3.1.
  cl_device_id get() const;

  /// Check if device is a host device
  ///
  /// \return true if SYCL device is a host device
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
  vector_class<device> create_sub_devices(size_t ComputeUnits) const;

  /// Partition device into sub devices
  ///
  /// Available only when prop is info::partition_property::partition_by_counts.
  /// If this SYCL device does not support
  /// info::partition_property::partition_by_counts a feature_not_supported
  /// exception must be thrown.
  ///
  /// \param Counts is a vector_class of desired compute units in sub devices.
  /// \return a vector_class of sub devices partitioned from this SYCL device by
  /// count sizes based on the Counts parameter.
  template <info::partition_property prop>
  vector_class<device>
  create_sub_devices(const vector_class<size_t> &Counts) const;

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
  vector_class<device>
  create_sub_devices(info::partition_affinity_domain AffinityDomain) const;

  /// Queries this SYCL device for information requested by the template
  /// parameter param
  ///
  /// Specializations of info::param_traits must be defined in accordance with
  /// the info parameters in Table 4.20 of SYCL Spec to facilitate returning the
  /// type associated with the param parameter.
  ///
  /// \return device info of type described in Table 4.20.
  template <info::device param>
  typename info::param_traits<info::device, param>::return_type
  get_info() const;

  /// Check SYCL extension support by device
  ///
  /// \param extension_name is a name of queried extension.
  /// \return true if SYCL device supports the extension.
  bool has_extension(const string_class &extension_name) const;

  /// Query available SYCL devices
  ///
  /// The returned vector_class must contain a single SYCL device
  /// that is a host device, permitted by the deviceType parameter
  ///
  /// \param deviceType is one of the values described in A.3 of SYCL Spec
  /// \return a vector_class containing all SYCL devices available in the system
  /// of the device type specified
  static vector_class<device>
  get_devices(info::device_type deviceType = info::device_type::all);

private:
  shared_ptr_class<detail::device_impl> impl;
  device(shared_ptr_class<detail::device_impl> impl) : impl(impl) {}

  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend
      typename std::add_pointer<typename decltype(T::impl)::element_type>::type
      detail::getRawSyclObjImpl(const T &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);
};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

namespace std {
template <> struct hash<cl::sycl::device> {
  size_t operator()(const cl::sycl::device &Device) const {
    return hash<cl::sycl::shared_ptr_class<cl::sycl::detail::device_impl>>()(
        cl::sycl::detail::getSyclObjImpl(Device));
  }
};
} // namespace std
