//==---------------- event_impl.hpp - SYCL event ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/event_info.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/stl.hpp>

#include <cassert>

namespace cl {
namespace sycl {
class context;
namespace detail {
class context_impl;
using ContextImplPtr = std::shared_ptr<cl::sycl::detail::context_impl>;
class queue_impl;

// Profiling info for the host execution.
class HostProfilingInfo {
  cl_ulong StartTime = 0;
  cl_ulong EndTime = 0;

public:
  cl_ulong getStartTime() const { return StartTime; }
  cl_ulong getEndTime() const { return EndTime; }

  void start();
  void end();
};

class event_impl {
public:
  event_impl() = default;
  event_impl(cl_event CLEvent, const context &SyclContext);
  event_impl(std::shared_ptr<cl::sycl::detail::queue_impl> Queue);

  // Threat all devices that don't support interoperability as host devices to
  // avoid attempts to call method get on such events.
  bool is_host() const;

  cl_event get() const;

  // Self is needed in order to pass shared_ptr to Scheduler.
  void wait(std::shared_ptr<cl::sycl::detail::event_impl> Self) const;

  void wait_and_throw(std::shared_ptr<cl::sycl::detail::event_impl> Self);

  template <info::event_profiling param>
  typename info::param_traits<info::event_profiling, param>::return_type
  get_profiling_info() const;

  template <info::event param>
  typename info::param_traits<info::event, param>::return_type get_info() const;

  ~event_impl();

  void waitInternal() const;

  void setComplete();

  // Warning. Returned reference will be invalid if event_impl was destroyed.
  RT::PiEvent &getHandleRef();
  const RT::PiEvent &getHandleRef() const;

  const ContextImplPtr &getContextImpl();

  // Warning. Provided cl_context inside ContextImplPtr must be associated
  // with the cl_event object stored in this class
  void setContextImpl(const ContextImplPtr &Context);

  void *getCommand() { return m_Command; }

  void setCommand(void *Command) { m_Command = Command; }

  HostProfilingInfo *getHostProfilingInfo() {
    return m_HostProfilingInfo.get();
  }

private:
  RT::PiEvent m_Event = nullptr;
  ContextImplPtr m_Context;
  bool m_OpenCLInterop = false;
  bool m_HostEvent = true;
  std::unique_ptr<HostProfilingInfo> m_HostProfilingInfo;
  void *m_Command = nullptr;
};

} // namespace detail
} // namespace sycl
} // namespace cl
