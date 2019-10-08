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
  virtual ~device_impl() = default;

  virtual cl_device_id get() const = 0;

  // Returns underlying native device object (if any) w/o reference count
  // modification. Caller must ensure the returned object lives on stack only.
  // It can also be safely passed to the underlying native runtime API.
  // Warning. Returned reference will be invalid if device_impl was destroyed.
  //
  virtual RT::PiDevice &getHandleRef() = 0;
  virtual const RT::PiDevice &getHandleRef() const = 0;

  virtual bool is_host() const = 0;

  virtual bool is_cpu() const = 0;

  virtual bool is_gpu() const = 0;

  virtual bool is_accelerator() const = 0;

  virtual platform get_platform() const = 0;

  virtual vector_class<device> create_sub_devices(size_t nbSubDev) const = 0;

  virtual vector_class<device>
  create_sub_devices(const vector_class<size_t> &counts) const = 0;

  virtual vector_class<device>
  create_sub_devices(info::partition_affinity_domain affinityDomain) const = 0;

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

  virtual bool has_extension(const string_class &extension_name) const = 0;
};

// TODO: 4.6.4 Partitioning into multiple SYCL devices
// TODO: 4.6.4.2 Device information descriptors
// TODO: Make code thread-safe
class device_impl_pi : public device_impl {
public:
  explicit device_impl_pi(RT::PiDevice a_device) : m_device(a_device) {
    // TODO catch an exception and put it to list of asynchronous exceptions
    PI_CALL(RT::piDeviceGetInfo(
      m_device, PI_DEVICE_INFO_TYPE, sizeof(RT::PiDeviceType), &m_type, 0));

    RT::PiDevice parent;
    // TODO catch an exception and put it to list of asynchronous exceptions
    PI_CALL(RT::piDeviceGetInfo(
      m_device, PI_DEVICE_INFO_PARENT, sizeof(RT::PiDevice), &parent, 0));

    m_isRootDevice = (nullptr == parent);
    if (!m_isRootDevice) {
      // TODO catch an exception and put it to list of asynchronous exceptions
      PI_CALL(RT::piDeviceRetain(m_device));
    }
  }

  ~device_impl_pi() {
    if (!m_isRootDevice) {
      // TODO catch an exception and put it to list of asynchronous exceptions
      CHECK_OCL_CODE_NO_EXC(RT::piDeviceRelease(m_device));
    }
  }

  cl_device_id get() const override {
    if (!m_isRootDevice) {
      // TODO catch an exception and put it to list of asynchronous exceptions
      PI_CALL(RT::piDeviceRetain(m_device));
    }
    // TODO: check that device is an OpenCL interop one
    return pi::cast<cl_device_id>(m_device);
  }

  RT::PiDevice &getHandleRef() override { return m_device; }
  const RT::PiDevice &getHandleRef() const override { return m_device; }

  bool is_host() const override { return false; }

  bool is_cpu() const override { return (m_type == PI_DEVICE_TYPE_CPU); }

  bool is_gpu() const override { return (m_type == PI_DEVICE_TYPE_GPU); }

  bool is_accelerator() const override {
    return (m_type == PI_DEVICE_TYPE_ACC);
  }

  platform get_platform() const override {
    RT::PiPlatform plt;
    // TODO catch an exception and put it to list of asynchronous exceptions
    PI_CALL(RT::piDeviceGetInfo(
      m_device, PI_DEVICE_INFO_PLATFORM, sizeof(plt), &plt, 0));

    // TODO: this possibly will violate common reference semantics,
    // particularly, equality comparison may fail for two consecutive
    // get_platform() on the same device, as it compares impl objects.
    return createSyclObjFromImpl<platform>(
      std::make_shared<platform_impl_pi>(plt));
  }

  bool has_extension(const string_class &extension_name) const override {
    string_class all_extension_names =
        get_device_info<string_class, info::device::extensions>::_(m_device);
    return (all_extension_names.find(extension_name) != std::string::npos);
  }

  vector_class<device>
  create_sub_devices(const cl_device_partition_property *Properties,
                     size_t SubDevicesCount) const;

  vector_class<device>
  create_sub_devices(size_t ComputeUnits) const override;

  vector_class<device>
  create_sub_devices(const vector_class<size_t> &Counts) const override;

  vector_class<device>
  create_sub_devices(info::partition_affinity_domain AffinityDomain) const override;

private:
  RT::PiDevice m_device = 0;
  RT::PiDeviceType m_type;
  bool m_isRootDevice = false;
}; // class device_impl_pi

// TODO: 4.6.4 Partitioning into multiple SYCL devices
// TODO: 4.6.4.2 Device information descriptors
// TODO: Make code thread-safe
class device_host : public device_impl {
public:
  device_host() = default;
  cl_device_id get() const override {
    throw invalid_object_error("This instance of device is a host instance");
  }
  RT::PiDevice &getHandleRef() override {
    throw invalid_object_error("This instance of device is a host instance");
  }
  const RT::PiDevice &getHandleRef() const override {
    throw invalid_object_error("This instance of device is a host instance");
  }

  bool is_host() const override { return true; }

  bool is_cpu() const override { return false; }

  bool is_gpu() const override { return false; }

  bool is_accelerator() const override { return false; }

  platform get_platform() const override { return platform(); }

  bool has_extension(const string_class &extension_name) const override {
    // TODO: implement extension management;
    return false;
  }

  vector_class<device> create_sub_devices(size_t nbSubDev) const override {
    // TODO: implement host device partitioning
    throw runtime_error(
        "Partitioning to subdevices of the host device is not implemented yet");
  }

  vector_class<device>
  create_sub_devices(const vector_class<size_t> &counts) const override {
    // TODO: implement host device partitioning
    throw runtime_error(
        "Partitioning to subdevices of the host device is not implemented yet");
  }

  vector_class<device>
  create_sub_devices(info::partition_affinity_domain affinityDomain) const override {
    // TODO: implement host device partitioning
    throw runtime_error(
        "Partitioning to subdevices of the host device is not implemented yet");
  }
}; // class device_host

} // namespace detail
} // namespace sycl
} // namespace cl
