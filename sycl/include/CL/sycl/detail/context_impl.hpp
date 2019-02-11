//==---------------- context_impl.hpp - SYCL context -----------*- C++-*---==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/platform.hpp>
#include <CL/sycl/stl.hpp>

#include <memory>
// 4.6.2 Context class

namespace cl {
namespace sycl {
// Forward declaration
class device;
namespace detail {
class context_impl {
public:
  context_impl(const device &Device, async_handler AsyncHandler);

  context_impl(const vector_class<cl::sycl::device> Devices,
               async_handler AsyncHandler);

  context_impl(cl_context ClContext, async_handler AsyncHandler);

  ~context_impl();

  cl_context get() const;

  bool is_host() const;

  platform get_platform() const;

  vector_class<device> get_devices() const;

  const async_handler &get_async_handler() const;

  template <info::context param>
  typename info::param_traits<info::context, param>::return_type
  get_info() const;

  // Warning. Returned reference will be invalid if context_impl was destroyed.
  cl_context &getHandleRef();

private:
  async_handler m_AsyncHandler;
  vector_class<device> m_Devices;
  cl_context m_ClContext;
  platform m_Platform;
  bool m_OpenCLInterop;
  bool m_HostContext;
};

} // namespace detail
} // namespace sycl
} // namespace cl
