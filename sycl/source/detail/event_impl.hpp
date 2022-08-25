//==---------------- event_impl.hpp - SYCL event ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/plugin.hpp>
#include <sycl/detail/cl.h>
#include <sycl/detail/common.hpp>
#include <sycl/detail/host_profiling_info.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/info/info_desc.hpp>
#include <sycl/stl.hpp>

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <optional>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
class context;
namespace detail {
class plugin;
class context_impl;
using ContextImplPtr = std::shared_ptr<sycl::detail::context_impl>;
class queue_impl;
using QueueImplPtr = std::shared_ptr<sycl::detail::queue_impl>;
class event_impl;
using EventImplPtr = std::shared_ptr<sycl::detail::event_impl>;

class event_impl {
public:
  enum HostEventState : int {
    HES_NotComplete = 0,
    HES_Complete,
    HES_Discarded
  };

  /// Constructs a ready SYCL event.
  ///
  /// If the constructed SYCL event is waited on it will complete immediately.
  /// Normally constructs a host event, use std::nullopt to instead instantiate
  /// a device event.
  event_impl(std::optional<HostEventState> State = HES_Complete)
      : MIsInitialized(false), MHostEvent(State), MIsFlushed(true),
        MState(State.value_or(HES_Complete)) {}

  /// Constructs an event instance from a plug-in event handle.
  ///
  /// The SyclContext must match the plug-in context associated with the
  /// ClEvent.
  ///
  /// \param Event is a valid instance of plug-in event.
  /// \param SyclContext is an instance of SYCL context.
  event_impl(RT::PiEvent Event, const context &SyclContext);
  event_impl(const QueueImplPtr &Queue);

  /// Checks if this event is a SYCL host event.
  ///
  /// All devices that do not support OpenCL interoperability are treated as
  /// host device to avoid attempts to call method get on such events.
  //
  /// \return true if this event is a SYCL host event.
  bool is_host();

  /// Waits for the event.
  ///
  /// Self is needed in order to pass shared_ptr to Scheduler.
  ///
  /// \param Self is a pointer to this event.
  void wait(std::shared_ptr<sycl::detail::event_impl> Self);

  /// Waits for the event.
  ///
  /// If any uncaught asynchronous errors occurred on the context that the
  /// event is waiting on executions from, then call that context's
  /// asynchronous error handler with those errors. Self is needed in order to
  /// pass shared_ptr to Scheduler.
  ///
  /// \param Self is a pointer to this event.
  void wait_and_throw(std::shared_ptr<sycl::detail::event_impl> Self);

  /// Clean up the command associated with the event. Assumes that the task this
  /// event is associated with has been completed.
  ///
  /// \param Self is a pointer to this event.
  void cleanupCommand(std::shared_ptr<sycl::detail::event_impl> Self) const;

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
  template <typename Param> typename Param::return_type get_profiling_info();

  /// Queries this SYCL event for information.
  ///
  /// \return depends on the information being requested.
  template <typename Param> typename Param::return_type get_info();

  ~event_impl();

  /// Waits for the event with respect to device type.
  void waitInternal();

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
  const plugin &getPlugin();

  /// Associate event with the context.
  ///
  /// Provided PiContext inside ContextImplPtr must be associated
  /// with the PiEvent object stored in this class
  ///
  /// @param Context is a shared pointer to an instance of valid context_impl.
  void setContextImpl(const ContextImplPtr &Context);

  /// Clear the event state
  void setStateIncomplete();

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
  pi_native_handle getNative();

  /// Returns vector of event dependencies.
  ///
  /// @return a reference to MPreparedDepsEvents.
  std::vector<std::shared_ptr<event_impl>> &getPreparedDepsEvents() {
    return MPreparedDepsEvents;
  }

  /// Returns vector of host event dependencies.
  ///
  /// @return a reference to MPreparedHostDepsEvents.
  std::vector<std::shared_ptr<event_impl>> &getPreparedHostDepsEvents() {
    return MPreparedHostDepsEvents;
  }

  /// Returns vector of event_impl that this event_impl depends on.
  ///
  /// @return a vector of "immediate" dependencies for this event_impl.
  std::vector<EventImplPtr> getWaitList();

  /// Performs a flush on the queue associated with this event if the user queue
  /// is different and the task associated with this event hasn't been submitted
  /// to the device yet.
  void flushIfNeeded(const QueueImplPtr &UserQueue);

  /// Cleans dependencies of this event_impl.
  void cleanupDependencyEvents();

  /// Cleans dependencies of this event's dependencies.
  void cleanDepEventsThroughOneLevel();

  /// Checks if this event is discarded by SYCL implementation.
  ///
  /// \return true if this event is discarded.
  bool isDiscarded() const { return MState == HES_Discarded; }

  void setNeedsCleanupAfterWait(bool NeedsCleanupAfterWait) {
    MNeedsCleanupAfterWait = NeedsCleanupAfterWait;
  }
  bool needsCleanupAfterWait() { return MNeedsCleanupAfterWait; }

  /// Checks if an event is in a fully intialized state. Default-constructed
  /// events will return true only after having initialized its native event,
  /// while other events will assume that they are fully initialized at
  /// construction, relying on external sources to supply member data.
  ///
  /// \return true if the event is considered to be in a fully initialized
  /// state.
  bool isInitialized() const noexcept { return MIsInitialized; }

private:
  // When instrumentation is enabled emits trace event for event wait begin and
  // returns the telemetry event generated for the wait
  void *instrumentationProlog(std::string &Name, int32_t StreamID,
                              uint64_t &instance_id) const;
  // Uses events generated by the Prolog and emits event wait done event
  void instrumentationEpilog(void *TelementryEvent, const std::string &Name,
                             int32_t StreamID, uint64_t IId) const;
  void checkProfilingPreconditions() const;
  // Events constructed without a context will lazily use the default context
  // when needed.
  void ensureContextInitialized();
  bool MIsInitialized = true;
  bool MIsContextInitialized = false;
  RT::PiEvent MEvent = nullptr;
  ContextImplPtr MContext;
  bool MHostEvent = true;
  std::unique_ptr<HostProfilingInfo> MHostProfilingInfo;
  void *MCommand = nullptr;
  std::weak_ptr<queue_impl> MQueue;
  const bool MIsProfilingEnabled = false;

  /// Dependency events prepared for waiting by backend.
  std::vector<EventImplPtr> MPreparedDepsEvents;
  std::vector<EventImplPtr> MPreparedHostDepsEvents;

  /// Indicates that the task associated with this event has been submitted by
  /// the queue to the device.
  std::atomic<bool> MIsFlushed = false;

  // State of host event. Employed only for host events and event with no
  // backend's representation (e.g. alloca). Used values are listed in
  // HostEventState enum.
  std::atomic<int> MState;

  // A temporary workaround for the current limitations of post enqueue graph
  // cleanup. Indicates that the command associated with this event isn't
  // handled by post enqueue cleanup yet and has to be deleted by cleanup after
  // wait.
  bool MNeedsCleanupAfterWait = false;

  std::mutex MMutex;
  std::condition_variable cv;

  friend std::vector<RT::PiEvent>
  getOrWaitEvents(std::vector<sycl::event> DepEvents,
                  std::shared_ptr<sycl::detail::context_impl> Context);
};

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
