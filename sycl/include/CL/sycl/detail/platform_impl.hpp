//==-------------- platform_impl.hpp - SYCL platform -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/force_device.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/detail/platform_info.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/stl.hpp>

// 4.6.2 Platform class
namespace cl {
namespace sycl {

// Forward declaration
class device_selector;
class device;

namespace detail {

class platform_impl {
public:
  platform_impl() = default;

  explicit platform_impl(const device_selector &);

  virtual bool has_extension(const string_class &extension_name) const = 0;

  virtual vector_class<device>
      get_devices(info::device_type = info::device_type::all) const = 0;

  template <info::platform param>
  typename info::param_traits<info::platform, param>::return_type
  get_info() const {
    if (is_host()) {
      return get_platform_info_host<param>();
    }
    return get_platform_info<
        typename info::param_traits<info::platform, param>::return_type,
        param>::_(this->getHandleRef());
  }

  virtual bool is_host() const = 0;

  virtual cl_platform_id get() const = 0;

  // Returns underlying native platform object.
  virtual const RT::PiPlatform &getHandleRef() const = 0;

  virtual ~platform_impl() = default;
};

// TODO: merge platform_impl_pi, platform_impl_host and platform_impl?
class platform_impl_pi : public platform_impl {
public:
  platform_impl_pi(RT::PiPlatform a_platform) : m_platform(a_platform) {}

  vector_class<device> get_devices(
      info::device_type deviceType = info::device_type::all) const override;

  bool has_extension(const string_class &extension_name) const override {
    string_class all_extension_names =
        get_platform_info<string_class, info::platform::extensions>::_(m_platform);
    return (all_extension_names.find(extension_name) != std::string::npos);
  }

  cl_platform_id get() const override { return pi::cast<cl_platform_id>(m_platform); }

  const RT::PiPlatform &getHandleRef() const override { return m_platform; }

  bool is_host() const override { return false; }

  static vector_class<platform> get_platforms();

private:
  RT::PiPlatform m_platform = 0;
}; // class platform_opencl

// TODO: implement extension management
// TODO: implement parameters treatment
// TODO: merge platform_impl_pi, platform_impl_host and platform_impl?
class platform_impl_host : public platform_impl {
public:
  vector_class<device> get_devices(
      info::device_type dev_type = info::device_type::all) const override;

  bool has_extension(const string_class &extension_name) const override {
    return false;
  }

  cl_platform_id get() const override {
    throw invalid_object_error("This instance of platform is a host instance");
  }
  const RT::PiPlatform &getHandleRef() const override {
    throw invalid_object_error("This instance of platform is a host instance");
  }

  bool is_host() const override { return true; }
}; // class platform_host


} // namespace detail
} // namespace sycl
} // namespace cl
