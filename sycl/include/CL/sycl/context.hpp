//==---------------- context.hpp - SYCL context ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/exception_list.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/detail/context_impl.hpp>
#include <CL/sycl/stl.hpp>
#include <type_traits>
// 4.6.2 Context class

namespace cl {
namespace sycl {
// Forward declarations
class device;
class platform;

class context {
public:
  explicit context(const async_handler &AsyncHandler = {});

  context(const device &Device, async_handler AsyncHandler = {});

  context(const platform &Platform, async_handler AsyncHandler = {});

  context(const vector_class<device> &DeviceList,
          async_handler AsyncHandler = {});

  context(cl_context ClContext, async_handler AsyncHandler = {});

  template <info::context param>
  typename info::param_traits<info::context, param>::return_type
  get_info() const;

  context(const context &rhs) = default;

  context(context &&rhs) = default;

  context &operator=(const context &rhs) = default;

  context &operator=(context &&rhs) = default;

  bool operator==(const context &rhs) const { return impl == rhs.impl; }

  bool operator!=(const context &rhs) const { return !(*this == rhs); }

  cl_context get() const;

  bool is_host() const;

  platform get_platform() const;

  vector_class<device> get_devices() const;

private:
  shared_ptr_class<detail::context_impl> impl;
  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend
      typename std::add_pointer<typename decltype(T::impl)::element_type>::type
      detail::getRawSyclObjImpl(const T &SyclObject);
};

} // namespace sycl
} // namespace cl

namespace std {
template <> struct hash<cl::sycl::context> {
  size_t operator()(const cl::sycl::context &Context) const {
    return hash<cl::sycl::shared_ptr_class<cl::sycl::detail::context_impl>>()(
        cl::sycl::detail::getSyclObjImpl(Context));
  }
};
} // namespace std
