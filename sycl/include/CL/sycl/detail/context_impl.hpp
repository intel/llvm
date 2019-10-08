//==---------------- context_impl.hpp - SYCL context -----------*- C++-*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/device_impl.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/detail/usm_dispatch.hpp>
#include <CL/sycl/exception_list.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/platform.hpp>
#include <CL/sycl/stl.hpp>

#include <map>
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

  // Returns underlying native context object (if any) w/o reference count
  // modification. Caller must ensure the returned object lives on stack only.
  // It can also be safely passed to the underlying native runtime API.
  // Warning. Returned reference will be invalid if context_impl was destroyed.
  RT::PiContext &getHandleRef();
  const RT::PiContext &getHandleRef() const;

  std::map<OSModuleHandle, RT::PiProgram> &getCachedPrograms() {
    return m_CachedPrograms;
  }
  std::map<RT::PiProgram, std::map<string_class, RT::PiKernel>> &
  getCachedKernels() {
    return m_CachedKernels;
  }

  std::shared_ptr<usm::USMDispatcher> getUSMDispatch() const;
private:
  async_handler m_AsyncHandler;
  vector_class<device> m_Devices;
  RT::PiContext m_Context;
  platform m_Platform;
  bool m_OpenCLInterop;
  bool m_HostContext;
  std::map<OSModuleHandle, RT::PiProgram> m_CachedPrograms;
  std::map<RT::PiProgram, std::map<string_class, RT::PiKernel>> m_CachedKernels;
  std::shared_ptr<usm::USMDispatcher> m_USMDispatch;
};

} // namespace detail
} // namespace sycl
} // namespace cl
