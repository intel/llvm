//==---------------- event_impl.hpp - SYCL event ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/adapter_impl.hpp>
#include <detail/helpers.hpp>
#include <sycl/detail/cl.h>
#include <sycl/detail/common.hpp>
#include <sycl/detail/host_profiling_info.hpp>
#include <sycl/detail/ur.hpp>
#include <sycl/event.hpp>
#include <sycl/info/info_desc.hpp>

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <optional>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental::detail {
class graph_impl;
}
class context;
namespace detail {
class adapter_impl;
class context_impl;
class queue_impl;
class event_impl;
using EventImplPtr = std::shared_ptr<sycl::detail::event_impl>;
class Command;

class event_impl {
  struct private_tag {
    explicit private_tag() = default;
  };

public:
  enum HostEventState : int {
    HES_NotComplete = 0,
    HES_Complete,
    HES_Discarded
  };

  /// Constructs a ready SYCL event.
  ///
  /// If the constructed SYCL event is waited on it will complete immediately.
  event_impl(private_tag)
      : MIsFlushed(true), MState(HES_Complete), MIsDefaultConstructed(true),
        MIsHostEvent(false) {
    // Need to fail in event() constructor  if there are problems with the
    // ONEAPI_DEVICE_SELECTOR. Deferring may lead to conficts with noexcept
    // event methods. This ::get() call uses static vars to read and parse the
    // ODS env var exactly once.
    SYCLConfig<ONEAPI_DEVICE_SELECTOR>::get();
  }

  /// Constructs an event instance from a UR event handle.
  ///
  /// The SyclContext must match the UR context associated with the
  /// ur_event_handle_t.
  ///
  /// \param Event is a valid instance of UR event.
  /// \param SyclContext is an instance of SYCL context.
  event_impl(ur_event_handle_t Event, const context &SyclContext, private_tag);

  event_impl(queue_impl &Queue, private_tag);
  event_impl(HostEventState State, private_tag);

  // Corresponds to `sycl::event{}`.
  static std::shared_ptr<event_impl> create_default_event() {
    return std::make_shared<event_impl>(private_tag{});
  }

  static std::shared_ptr<event_impl>
  create_from_handle(ur_event_handle_t Event, const context &SyclContext) {
    return std::make_shared<event_impl>(Event, SyclContext, private_tag{});
  }

  static std::shared_ptr<event_impl> create_device_event(queue_impl &queue) {
    return std::make_shared<event_impl>(queue, private_tag{});
  }

  static std::shared_ptr<event_impl> create_discarded_event() {
    return std::make_shared<event_impl>(HostEventState::HES_Discarded,
                                        private_tag{});
  }

  static std::shared_ptr<event_impl> create_completed_host_event() {
    return std::make_shared<event_impl>(HostEventState::HES_Complete,
                                        private_tag{});
  }

  static std::shared_ptr<event_impl> create_incomplete_host_event() {
    return std::make_shared<event_impl>(HostEventState::HES_NotComplete,
                                        private_tag{});
  }

  /// Sets a queue associated with the event
  ///
  /// Please note that this function changes the event state
  /// as it was constructed with the queue based constructor.
  ///
  /// \param Queue is a queue to be associated with the event
  void setQueue(queue_impl &Queue);

  /// Waits for the event.
  ///
  /// \param Success is an optional parameter that, when set to a non-null
  ///        pointer, indicates that failure is a valid outcome for this wait
  ///        (e.g., in case of a non-blocking read from a pipe), and the value
  ///        it's pointing to is then set according to the outcome.
  void wait(bool *Success = nullptr);

  /// Waits for the event.
  ///
  /// If any uncaught asynchronous errors occurred on the context that the
  /// event is waiting on executions from, then call that context's
  /// asynchronous error handler with those errors. Self is needed in order to
  /// pass shared_ptr to Scheduler.
  void wait_and_throw();

  /// Queries this event for profiling information.
  ///
  /// If the requested info is not available when this member function is
  /// called due to incompletion of command groups associated with the event,
  /// then the call to this member function will block until the requested
  /// info is available. If the queue which submitted the command group this
  /// event is associated with was not constructed with the
  /// property::queue::enable_profiling property, a SYCL exception with
  /// errc::invalid error code is thrown.
  ///
  /// \return depends on template parameter.
  template <typename Param> typename Param::return_type get_profiling_info();

  /// Queries this SYCL event for information.
  ///
  /// \return depends on the information being requested.
  template <typename Param> typename Param::return_type get_info();

  /// Queries this SYCL event for SYCL backend-specific information.
  ///
  /// \return depends on information being queried.
  template <typename Param>
  typename Param::return_type get_backend_info() const;

  ~event_impl();

  /// Waits for the event with respect to device type.
  /// \param Success is an optional parameter that, when set to a non-null
  ///        pointer, indicates that failure is a valid outcome for this wait
  ///        (e.g., in case of a non-blocking read from a pipe), and the value
  ///        it's pointing to is then set according to the outcome.
  void waitInternal(bool *Success = nullptr);

  /// Marks this event as completed.
  void setComplete();

  /// Returns raw interoperability event handle.
  ur_event_handle_t getHandle() const;

  /// Set event handle for this event object.
  void setHandle(const ur_event_handle_t &UREvent);

  /// Returns context that is associated with this event.
  context_impl &getContextImpl();

  /// \return the Adapter associated with the context of this event.
  /// Should be called when this is not a Host Event.
  adapter_impl &getAdapter();

  /// Associate event with the context.
  ///
  /// Provided UrContext inside Context must be associated
  /// with the UrEvent object stored in this class
  void setContextImpl(context_impl &Context);

  /// Clear the event state
  void setStateIncomplete();

  /// Set state as discarded.
  void setStateDiscarded() { MState = HES_Discarded; }

  /// Returns command that is associated with the event.
  ///
  /// Scheduler mutex must be locked in read mode when this is called.
  ///
  /// @return a generic pointer to Command object instance.
  Command *getCommand() { return MCommand; }

  /// Associates this event with the command.
  ///
  /// Scheduler mutex must be locked in write mode when this is called.
  ///
  /// @param Command is a generic pointer to Command object instance.
  void setCommand(Command *Cmd);

  /// Returns host profiling information.
  ///
  /// @return a pointer to HostProfilingInfo instance.
  HostProfilingInfo *getHostProfilingInfo() { return MHostProfilingInfo.get(); }

  /// Gets the native handle of the SYCL event.
  ///
  /// \return a native handle.
  ur_native_handle_t getNative();

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
  void flushIfNeeded(queue_impl *UserQueue);

  /// Cleans dependencies of this event_impl.
  void cleanupDependencyEvents();

  /// Cleans dependencies of this event's dependencies.
  void cleanDepEventsThroughOneLevel();

  /// Cleans dependencies of this event's dependencies w/o locking MMutex.
  void cleanDepEventsThroughOneLevelUnlocked();

  /// Checks if this event is discarded by SYCL implementation.
  ///
  /// \return true if this event is discarded.
  bool isDiscarded() const { return MState == HES_Discarded; }

  /// Returns worker queue for command.
  ///
  /// @return shared_ptr to MWorkerQueue, please be aware it can be empty
  /// pointer
  std::shared_ptr<sycl::detail::queue_impl> getWorkerQueue() {
    return MWorkerQueue.lock();
  };

  /// Sets worker queue for command.
  ///
  /// @return
  void setWorkerQueue(std::weak_ptr<queue_impl> WorkerQueue) {
    MWorkerQueue = std::move(WorkerQueue);
  };

  /// Sets original queue used for submission.
  ///
  /// @return
  void setSubmittedQueue(std::weak_ptr<queue_impl> SubmittedQueue);

  /// Indicates if this event is not associated with any command and doesn't
  /// have native handle.
  ///
  /// @return true if no associated command and no event handle.
  bool isNOP() { return !MCommand && !getHandle(); }

  /// Calling this function queries the current device timestamp and sets it as
  /// submission time for the command associated with this event.
  void setSubmissionTime();

  /// @return Submission time for command associated with this event
  uint64_t getSubmissionTime();

  std::shared_ptr<sycl::detail::queue_impl> getSubmittedQueue() const {
    return MSubmittedQueue.lock();
  };

  /// Checks if this event is complete.
  ///
  /// \return true if this event is complete.
  bool isCompleted();

  /// Checks if associated command is enqueued
  ///
  /// \return true if command passed enqueue
  bool isEnqueued() const noexcept { return MIsEnqueued; };

  void attachEventToComplete(const EventImplPtr &Event) {
    std::lock_guard<std::mutex> Lock(MMutex);
    MPostCompleteEvents.push_back(Event);
  }

  void attachEventToCompleteWeak(const std::weak_ptr<event_impl> &Event) {
    std::lock_guard<std::mutex> Lock(MMutex);
    MWeakPostCompleteEvents.push_back(Event);
  }

  bool isDefaultConstructed() const noexcept { return MIsDefaultConstructed; }

  // Sets a sync point which is used when this event represents an enqueue to a
  // Command Buffer.
  void setSyncPoint(ur_exp_command_buffer_sync_point_t SyncPoint) {
    MSyncPoint = SyncPoint;
  }

  // Get the sync point associated with this event.
  ur_exp_command_buffer_sync_point_t getSyncPoint() const { return MSyncPoint; }

  void setCommandGraph(
      std::shared_ptr<ext::oneapi::experimental::detail::graph_impl> Graph) {
    MGraph = Graph;
  }

  std::shared_ptr<ext::oneapi::experimental::detail::graph_impl>
  getCommandGraph() const {
    return MGraph.lock();
  }

  bool hasCommandGraph() const { return !MGraph.expired(); }

  void setEventFromSubmittedExecCommandBuffer(bool value) {
    MEventFromSubmittedExecCommandBuffer = value;
  }

  bool isEventFromSubmittedExecCommandBuffer() const {
    return MEventFromSubmittedExecCommandBuffer;
  }

  void setProfilingEnabled(bool Value) { MIsProfilingEnabled = Value; }

  // Sets a command-buffer command when this event represents an enqueue to a
  // Command Buffer.
  void setCommandBufferCommand(ur_exp_command_buffer_command_handle_t Command) {
    MCommandBufferCommand = Command;
  }

  ur_exp_command_buffer_command_handle_t getCommandBufferCommand() const {
    return MCommandBufferCommand;
  }

  const std::vector<EventImplPtr> &getPostCompleteEvents() const {
    return MPostCompleteEvents;
  }

  void setEnqueued() { MIsEnqueued = true; }

  bool isHost() { return MIsHostEvent; }

  void markAsProfilingTagEvent() { MProfilingTagEvent = true; }

  bool isProfilingTagEvent() const noexcept { return MProfilingTagEvent; }

  // Check if this event is an interoperability event.
  bool isInterop() const noexcept {
    // As an indication of interoperability event, we use the absence of the
    // queue and command, as well as the fact that it is not in enqueued state.
    return MEvent && MQueue.expired() && !MIsEnqueued && !MCommand;
  }

  // Initializes the host profiling info for the event.
  void initHostProfilingInfo();

protected:
  // When instrumentation is enabled emits trace event for event wait begin and
  // returns the telemetry event generated for the wait
#ifdef XPTI_ENABLE_INSTRUMENTATION
  void *instrumentationProlog(std::string &Name, xpti::stream_id_t StreamID,
                              uint64_t &instance_id) const;
  // Uses events generated by the Prolog and emits event wait done event
  void instrumentationEpilog(void *TelementryEvent, const std::string &Name,
                             xpti::stream_id_t StreamID, uint64_t IId) const;
#endif
  void checkProfilingPreconditions() const;

  std::atomic<ur_event_handle_t> MEvent = nullptr;
  // Stores submission time of command associated with event
  uint64_t MSubmitTime = 0;
  std::shared_ptr<context_impl> MContext;
  std::unique_ptr<HostProfilingInfo> MHostProfilingInfo;
  Command *MCommand = nullptr;
  std::weak_ptr<queue_impl> MQueue;
  bool MIsProfilingEnabled = false;

  std::weak_ptr<queue_impl> MWorkerQueue;
  std::weak_ptr<queue_impl> MSubmittedQueue;

  /// Dependency events prepared for waiting by backend.
  std::vector<EventImplPtr> MPreparedDepsEvents;
  std::vector<EventImplPtr> MPreparedHostDepsEvents;

  std::vector<EventImplPtr> MPostCompleteEvents;
  // short term WA for stream:
  // MPostCompleteEvents is split into two storages now. Original storage is
  // used by graph extension and represents backward links.
  // MWeakPostCompleteEvents represents weak forward references (used in stream
  // only). Used only for host tasks now since they do not support post enqueue
  // cleanup and event == nullptr could happen only when host task is completed
  // (and Command that holding reference to its event is deleted). TO DO: to
  // eliminate forward references from stream implementation and remove this
  // storage.
  std::vector<std::weak_ptr<event_impl>> MWeakPostCompleteEvents;

  /// Indicates that the task associated with this event has been submitted by
  /// the queue to the device.
  std::atomic<bool> MIsFlushed = false;

  // State of host event. Employed only for host events and event with no
  // backend's representation (e.g. alloca). Used values are listed in
  // HostEventState enum.
  std::atomic<int> MState;

  std::mutex MMutex;
  std::condition_variable cv;

  /// Store the command graph associated with this event, if any.
  /// This event is also be stored in the graph so a weak_ptr is used.
  std::weak_ptr<ext::oneapi::experimental::detail::graph_impl> MGraph;
  /// Indicates that the event results from a command graph submission.
  bool MEventFromSubmittedExecCommandBuffer = false;

  // If this event represents a submission to a
  // ur_exp_command_buffer_sync_point_t the sync point for that submission is
  // stored here.
  ur_exp_command_buffer_sync_point_t MSyncPoint = 0;

  // If this event represents a submission to a
  // ur_exp_command_buffer_command_handle_t the command-buffer command
  // (if any) associated with that submission is stored here.
  ur_exp_command_buffer_command_handle_t MCommandBufferCommand = nullptr;

  // Signifies whether this event is the result of a profiling tag command. This
  // allows for profiling, even if the queue does not have profiling enabled.
  bool MProfilingTagEvent = false;

  std::atomic_bool MIsEnqueued{false};

  // Events constructed without a context will lazily use the default context
  // when needed.
  void initContextIfNeeded();
  // Event class represents 3 different kinds of operations:
  // | type  | has UR event | MContext | MIsHostTask | MIsDefaultConstructed |
  // | dev   | true         | !nullptr | false       | false                 |
  // | host  | false        | nullptr  | true        | false                 |
  // |default|   *          |    *     | false       | true                  |
  // Default constructed event is created with empty ctor in host code, MContext
  // is lazily initialized with default device context on first context query.
  // MEvent is lazily created in first ur handle query.
  bool MIsDefaultConstructed = false;
  bool MIsHostEvent = false;
};

using events_iterator =
    variadic_iterator<event,
                      std::vector<std::shared_ptr<event_impl>>::const_iterator,
                      std::vector<event>::const_iterator,
                      std::vector<event_impl *>::const_iterator, event_impl *>;

class events_range : public iterator_range<events_iterator> {
private:
  using Base = iterator_range<events_iterator>;

public:
  using Base::Base;
};
} // namespace detail
} // namespace _V1
} // namespace sycl
