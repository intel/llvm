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
#include <algorithm>
#include <memory>

namespace cl {
namespace sycl {

// Forward declaration
class platform;

namespace detail {
// TODO: 4.6.4 Partitioning into multiple SYCL devices
// TODO: 4.6.4.2 Device information descriptors
// TODO: Make code thread-safe
class device_impl {
public:
  device_impl();
  explicit device_impl(RT::PiDevice Device) : MDevice(Device) {
    // TODO catch an exception and put it to list of asynchronous exceptions
    PI_CALL(RT::piDeviceGetInfo(
      MDevice, PI_DEVICE_INFO_TYPE, sizeof(RT::PiDeviceType), &MType, 0));

    RT::PiDevice parent;
    // TODO catch an exception and put it to list of asynchronous exceptions
    PI_CALL(RT::piDeviceGetInfo(
      MDevice, PI_DEVICE_INFO_PARENT, sizeof(RT::PiDevice), &parent, 0));

    MIsRootDevice = (nullptr == parent);
    if (!MIsRootDevice) {
      // TODO catch an exception and put it to list of asynchronous exceptions
      PI_CALL(RT::piDeviceRetain(MDevice));
    }
  }
  ~device_impl() {
    if (!MIsRootDevice && MIsHostDevice) {
      // TODO catch an exception and put it to list of asynchronous exceptions
      CHECK_OCL_CODE_NO_EXC(RT::piDeviceRelease(MDevice));
    }
  }

  cl_device_id get() const {
    if (MIsHostDevice)
      throw invalid_object_error("This instance of device is a host instance");

    if (!MIsRootDevice) {
      // TODO catch an exception and put it to list of asynchronous exceptions
      PI_CALL(RT::piDeviceRetain(MDevice));
    }
    // TODO: check that device is an OpenCL interop one
    return pi::cast<cl_device_id>(MDevice);
  }

  // Returns underlying native device object (if any) w/o reference count
  // modification. Caller must ensure the returned object lives on stack only.
  // It can also be safely passed to the underlying native runtime API.
  // Warning. Returned reference will be invalid if device_impl was destroyed.
  //
  RT::PiDevice &getHandleRef() {
    if (MIsHostDevice)
      throw invalid_object_error("This instance of device is a host instance");
    return MDevice;
  }
  const RT::PiDevice &getHandleRef() const {
    if (MIsHostDevice)
      throw invalid_object_error("This instance of device is a host instance");
    return MDevice;
  }

  bool is_host() const { return MIsHostDevice; }

  bool is_cpu() const { return (MType == PI_DEVICE_TYPE_CPU); }

  bool is_gpu() const { return (MType == PI_DEVICE_TYPE_GPU); }

  bool is_accelerator() const {
    return (MType == PI_DEVICE_TYPE_ACC);
  }

  platform get_platform() const {
    if (MIsHostDevice)
      return platform();

    RT::PiPlatform plt;
    // TODO catch an exception and put it to list of asynchronous exceptions
    PI_CALL(RT::piDeviceGetInfo(
      MDevice, PI_DEVICE_INFO_PLATFORM, sizeof(plt), &plt, 0));

    // TODO: this possibly will violate common reference semantics,
    // particularly, equality comparison may fail for two consecutive
    // get_platform() on the same device, as it compares impl objects.
    return createSyclObjFromImpl<platform>(
      std::make_shared<platform_impl_pi>(plt));
  }

  vector_class<device> create_sub_devices(size_t nbSubDev) const;

  vector_class<device>
  create_sub_devices(const vector_class<size_t> &counts) const;

  vector_class<device>
  create_sub_devices(info::partition_affinity_domain affinityDomain) const;

  static vector_class<device>
  get_devices(info::device_type deviceType = info::device_type::all);

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

  bool is_partition_supported(info::partition_property Prop) const {
    auto SupportedProperties = get_info<info::device::partition_properties>();
    return std::find(SupportedProperties.begin(), SupportedProperties.end(),
                     Prop) != SupportedProperties.end();
  }

  bool
  is_affinity_supported(info::partition_affinity_domain AffinityDomain) const {
    auto SupportedDomains =
        get_info<info::device::partition_affinity_domains>();
    return std::find(SupportedDomains.begin(), SupportedDomains.end(),
                     AffinityDomain) != SupportedDomains.end();
  }

  bool has_extension(const string_class &extension_name) const {
    if (MIsHostDevice)
      return false;

    string_class all_extension_names =
        get_device_info<string_class, info::device::extensions>::_(MDevice);
    return (all_extension_names.find(extension_name) != std::string::npos);
  }

  vector_class<device>
  create_sub_devices(const cl_device_partition_property *Properties,
                     size_t SubDevicesCount) const;
private:
  RT::PiDevice MDevice = 0;
  RT::PiDeviceType MType;
  bool MIsRootDevice = false;
  bool MIsHostDevice;
};

// TODO: 4.6.4 Partitioning into multiple SYCL devices
// TODO: 4.6.4.2 Device information descriptors
// TODO: Make code thread-safe
class device_impl_pi : public device_impl {
public:









}; // class device_impl
} // namespace detail
} // namespace sycl
} // namespace cl
