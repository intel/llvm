//==---------------- platform.hpp - SYCL platform --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/platform_impl.hpp>
#include <CL/sycl/detail/platform_info.hpp>
#include <CL/sycl/stl.hpp>
// 4.6.2 Platform class
#include <memory>
#include <utility>
namespace cl {
namespace sycl {
// TODO: make code thread-safe

// Forward declaration
class device_selector;
class device;

class platform {
public:
  platform();

  explicit platform(cl_platform_id platform_id);

  explicit platform(const device_selector &);

  template <info::platform param>
  typename info::param_traits<info::platform, param>::return_type
  get_info() const {
    return impl->get_info<param>();
  }

  platform(const platform &rhs) = default;

  platform(platform &&rhs) = default;

  platform &operator=(const platform &rhs) = default;

  platform &operator=(platform &&rhs) = default;

  bool operator==(const platform &rhs) const { return impl == rhs.impl; }

  bool operator!=(const platform &rhs) const { return !(*this == rhs); }

  cl_platform_id get() const { return impl->get(); }

  bool has_extension(const string_class &extension_name) const {
    return impl->has_extension(extension_name);
  }

  bool is_host() const { return impl->is_host(); }

  vector_class<device>
  get_devices(info::device_type dev_type = info::device_type::all) const;

  static vector_class<platform> get_platforms();

private:
  std::shared_ptr<detail::platform_impl> impl;
  platform(std::shared_ptr<detail::platform_impl> impl) : impl(impl) {}

  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);
  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);

}; // class platform
} // namespace sycl
} // namespace cl

namespace std {
template <> struct hash<cl::sycl::platform> {
  size_t operator()(const cl::sycl::platform &p) const {
    return hash<std::shared_ptr<cl::sycl::detail::platform_impl>>()(
        cl::sycl::detail::getSyclObjImpl(p));
  }
};
} // namespace std
