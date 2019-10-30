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

__SYCL_INLINE namespace cl {
namespace sycl {
class context;
namespace detail {
class context_impl;
using ContextImplPtr = std::shared_ptr<cl::sycl::detail::context_impl>;
class queue_impl;
using QueueImplPtr = std::shared_ptr<cl::sycl::detail::queue_impl>;

/// Profiling info for the host execution.
class HostProfilingInfo {
  cl_ulong StartTime = 0;
  cl_ulong EndTime = 0;

public:
  /// Returns event's start time.
  ///
  /// @return event's start time in nanoseconds.
  cl_ulong getStartTime() const { return StartTime; }
  /// Returns event's end time.
  ///
  /// @return event's end time in nanoseconds.
  cl_ulong getEndTime() const { return EndTime; }

  /// Measures event's start time.
  void start();
  /// Measures event's end time.
  void end();
};

class event_impl {
public:
  /// Constructs a ready SYCL event.
  ///
  /// If the constructed SYCL event is waited on it will complete immediately.
  event_impl() = default;
  /// Constructs an event instance from a plug-in event handle.
  ///
  /// The SyclContext must match the plug-in context associated with the ClEvent.
  ///
  /// @param Event is a valid instance of plug-in event.
  /// @param SyclContext is an instance of SYCL context.
  event_impl(RT::PiEvent Event, const context &SyclContext);
  event_impl(QueueImplPtr Queue);

  /// Checks if this event is a SYCL host event.
  ///
  /// All devices that do not support OpenCL interoperability are treated as
  /// host device to avoid attempts to call method get on such events.
  //
  /// @return true if this event is a SYCL host event.
  bool is_host() const;

  /// Returns a valid OpenCL event interoperability handle.
  ///
  /// @return a valid instance of OpenCL cl_event.
  cl_event get() const;

  /// Waits for the event.
  ///
  /// Self is needed in order to pass shared_ptr to Scheduler.
  ///
  /// @param Self is a pointer to this event.
  void wait(std::shared_ptr<cl::sycl::detail::event_impl> Self) const;

  /// Waits for the event.
  ///
  /// If any uncaught asynchronous errors occurred on the context that the event
  /// is waiting on executions from, then call that context's asynchronous error
  /// handler with those errors.
  /// Self is needed in order to pass shared_ptr to Scheduler.
  ///
  /// @param Self is a pointer to this event.
  void wait_and_throw(std::shared_ptr<cl::sycl::detail::event_impl> Self);

  /// Queries this event for profiling information.
  ///
  /// If the requested info is not available when this member function is called
  /// due to incompletion of command groups associated with the event, then the
  /// call to this member function will block until the requested info is
  /// available. If the queue which submitted the command group this event is
  /// associated with was not constructed with the
  /// property::queue::enable_profiling property, an invalid_object_error SYCL
  /// exception is thrown.
  ///
  /// @return depends on template parameter.
  template <info::event_profiling param>
  typename info::param_traits<info::event_profiling, param>::return_type
  get_profiling_info() const;

  /// Queries this SYCL event for information.
  ///
  /// @return depends on the information being requested.
  template <info::event param>
  typename info::param_traits<info::event, param>::return_type get_info() const;

  ~event_impl();

  /// Waits for the event with respect to device type.
  void waitInternal() const;

  /// Marks this event as completed.
  void setComplete();

  /// Returns raw interoperability event handle. Returned reference will be]
  /// invalid if event_impl was destroyed.
  ///
  /// @return a reference to an instance of plug-in event handle.
  RT::PiEvent &getHandleRef();
  /// Returns raw interoperability event handle. Returned reference will be]
  /// invalid if event_impl was destroyed.
  ///
  /// @return a const reference to an instance of plug-in event handle.
  const RT::PiEvent &getHandleRef() const;

  /// Returns context that is associated with this event.
  ///
  /// @return a shared pointer to a valid context_impl.
  const ContextImplPtr &getContextImpl();

  /// Associate event with the context.
  ///
  /// Provided PiContext inside ContextImplPtr must be associated
  /// with the PiEvent object stored in this class
  ///
  /// @param Context is a shared pointer to an instance of valid context_impl.
  void setContextImpl(const ContextImplPtr &Context);

  /// Returns command that is associated with the event.
  ///
  /// @return a generic pointer to Command object instance.
  void *getCommand() { return MCommand; }

  /// Associates this event with the command.
  ///
  /// @param Command is a generic pointer to Command object instance.
  void setCommand(void *Command) { MCommand = Command; }

  /// Returns host profiling information.
  ///
  /// @return a pointer to HostProfilingInfo instance.
  HostProfilingInfo *getHostProfilingInfo() { return MHostProfilingInfo.get(); }

private:
  RT::PiEvent MEvent = nullptr;
  ContextImplPtr MContext;
  QueueImplPtr MQueue;
  bool MOpenCLInterop = false;
  bool MHostEvent = true;
  std::unique_ptr<HostProfilingInfo> MHostProfilingInfo;
  void *MCommand = nullptr;
};

} // namespace detail
} // namespace sycl
} // namespace cl
