//==------------------- device.hpp - SYCL device ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/aspects.hpp>
#include <CL/sycl/detail/backend_traits.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/export.hpp>
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
auto getDeviceComparisonLambda();
}

// A tag to allow for creating a host device
struct host_device {};

/// The SYCL device class encapsulates a single SYCL device on which kernels
/// may be executed.
///
/// \ingroup sycl_api
class __SYCL_EXPORT device {
public:
  /// Constructs a SYCL device instance as a host device.
  device(host_device);

  /// Constructs a SYCL device instance as selected by default_selector
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
#ifdef __SYCL_INTERNAL_API
  cl_device_id get() const;
#endif

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

  /// Gets the native handle of the SYCL device.
  ///
  /// \return a native handle, the type of which defined by the backend.
  template <backend Backend>
  __SYCL_DEPRECATED("Use SYCL 2020 sycl::get_native free function")
  backend_return_t<Backend, device> get_native() const {
    // In CUDA CUdevice isn't an opaque pointer, unlike a lot of the others,
    // but instead a 32-bit int (on all relevant systems). Different
    // backends use the same function for this purpose so static_cast is
    // needed in some cases but not others, so a C-style cast was chosen.
    return (backend_return_t<Backend, device>)getNative();
  }

  /// Indicates if the SYCL device has the given feature.
  ///
  /// \param Aspect is one of the values in Table 4.20 of the SYCL 2020
  /// Provisional Spec.
  ///
  /// \return true if the SYCL device has the given feature.
  bool has(aspect Aspect) const;

private:
  std::shared_ptr<detail::device_impl> impl;
  device(std::shared_ptr<detail::device_impl> impl) : impl(impl) {}

  pi_native_handle getNative() const;

  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend
      typename detail::add_pointer_t<typename decltype(T::impl)::element_type>
      detail::getRawSyclObjImpl(const T &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);

  friend auto detail::getDeviceComparisonLambda();
};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

namespace std {
template <> struct hash<cl::sycl::device> {
  size_t operator()(const cl::sycl::device &Device) const {
    return hash<std::shared_ptr<cl::sycl::detail::device_impl>>()(
        cl::sycl::detail::getSyclObjImpl(Device));
  }
};
} // namespace std
