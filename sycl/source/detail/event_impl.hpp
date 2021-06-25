//==---------------- event_impl.hpp - SYCL event ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/host_profiling_info.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/stl.hpp>
#include <detail/plugin.hpp>

#include <atomic>
#include <cassert>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
class context;
namespace detail {
class plugin;
class context_impl;
using ContextImplPtr = std::shared_ptr<cl::sycl::detail::context_impl>;
class queue_impl;
using QueueImplPtr = std::shared_ptr<cl::sycl::detail::queue_impl>;

class event_impl {
public:
  /// Constructs a ready SYCL event.
  ///
  /// If the constructed SYCL event is waited on it will complete immediately.
  event_impl();
  /// Constructs an event instance from a plug-in event handle.
  ///
  /// The SyclContext must match the plug-in context associated with the
  /// ClEvent.
  ///
  /// \param Event is a valid instance of plug-in event.
  /// \param SyclContext is an instance of SYCL context.
  event_impl(RT::PiEvent Event, const context &SyclContext);
  event_impl(QueueImplPtr Queue);

  /// Checks if this event is a SYCL host event.
  ///
  /// All devices that do not support OpenCL interoperability are treated as
  /// host device to avoid attempts to call method get on such events.
  //
  /// \return true if this event is a SYCL host event.
  bool is_host() const;

  /// Returns a valid OpenCL event interoperability handle.
  ///
  /// \return a valid instance of OpenCL cl_event.
  cl_event get() const;

  /// Waits for the event.
  ///
  /// Self is needed in order to pass shared_ptr to Scheduler.
  ///
  /// \param Self is a pointer to this event.
  void wait(std::shared_ptr<cl::sycl::detail::event_impl> Self) const;

  /// Waits for the event.
  ///
  /// If any uncaught asynchronous errors occurred on the context that the
  /// event is waiting on executions from, then call that context's
  /// asynchronous error handler with those errors. Self is needed in order to
  /// pass shared_ptr to Scheduler.
  ///
  /// \param Self is a pointer to this event.
  void wait_and_throw(std::shared_ptr<cl::sycl::detail::event_impl> Self);

  /// Queries this event for profiling information.
  ///
  /// If the requested info is not available when this member function is
  /// called due to incompletion of command groups associated with the event,
  /// then the call to this member function will block until the requested
  /// info is available. If the queue which submitted the command group this
  /// event is associated with was not constructed with the
  /// property::queue::enable_profiling property, an invalid_object_error SYCL
  /// exception is thrown.
  ///
  /// \return depends on template parameter.
  template <info::event_profiling param>
  typename info::param_traits<info::event_profiling, param>::return_type
  get_profiling_info() const;

  /// Queries this SYCL event for information.
  ///
  /// \return depends on the information being requested.
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
  /// \return a reference to an instance of plug-in event handle.
  RT::PiEvent &getHandleRef();
  /// Returns raw interoperability event handle. Returned reference will be]
  /// invalid if event_impl was destroyed.
  ///
  /// \return a const reference to an instance of plug-in event handle.
  const RT::PiEvent &getHandleRef() const;

  /// Returns context that is associated with this event.
  ///
  /// \return a shared pointer to a valid context_impl.
  const ContextImplPtr &getContextImpl();

  /// \return the Plugin associated with the context of this event.
  /// Should be called when this is not a Host Event.
  const plugin &getPlugin() const;

  /// Associate event with the context.
  ///
  /// Provided PiContext inside ContextImplPtr must be associated
  /// with the PiEvent object stored in this class
  ///
  /// @param Context is a shared pointer to an instance of valid context_impl.
  void setContextImpl(const ContextImplPtr &Context);

  /// Returns command that is associated with the event.
  ///
  /// Scheduler mutex must be locked in read mode when this is called.
  ///
  /// @return a generic pointer to Command object instance.
  void *getCommand() { return MCommand; }

  /// Associates this event with the command.
  ///
  /// Scheduler mutex must be locked in write mode when this is called.
  ///
  /// @param Command is a generic pointer to Command object instance.
  void setCommand(void *Command) { MCommand = Command; }

  /// Returns host profiling information.
  ///
  /// @return a pointer to HostProfilingInfo instance.
  HostProfilingInfo *getHostProfilingInfo() { return MHostProfilingInfo.get(); }

  /// Gets the native handle of the SYCL event.
  ///
  /// \return a native handle.
  pi_native_handle getNative() const;

private:
  // When instrumentation is enabled emits trace event for event wait begin and
  // returns the telemetry event generated for the wait
  void *instrumentationProlog(string_class &Name, int32_t StreamID,
                              uint64_t &instance_id) const;
  // Uses events generated by the Prolog and emits event wait done event
  void instrumentationEpilog(void *TelementryEvent, const string_class &Name,
                             int32_t StreamID, uint64_t IId) const;

  RT::PiEvent MEvent = nullptr;
  ContextImplPtr MContext;
  bool MOpenCLInterop = false;
  bool MHostEvent = true;
  std::unique_ptr<HostProfilingInfo> MHostProfilingInfo;
  void *MCommand = nullptr;

  enum HostEventState : int { HES_NotComplete = 0, HES_Complete };

  // State of host event. Employed only for host events and event with no
  // backend's representation (e.g. alloca). Used values are listed in
  // HostEventState enum.
  std::atomic<int> MState;
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
