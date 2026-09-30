//==---------------- event_binding.hpp - SYCL event binding ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/host_profiling_info.hpp>
#include <sycl/detail/ur.hpp>

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace sycl {
inline namespace _V1 {
namespace detail {

class adapter_impl;
class Command;
class event_binding;
class event_impl;
class queue_impl;

/// Completion state of a signal without a backend event (host events, alloca
/// and the like).
enum HostEventState : int { HES_NotComplete = 0, HES_Complete, HES_Discarded };

/// A dependency of a command on an event, captured when the command is
/// submitted.
///
/// The binding is the signal the event represented at submission; everything
/// which belongs to the signal - backend event, producing command, completion,
/// worker queue - is read from it, never from the event, so that the dependency
/// stays the same if the event is enqueued for signaling again before the
/// command reaches the backend. The event is kept for what belongs to the
/// event rather than to the signal (event::get_wait_list, the kind of event,
/// its context) and to keep it alive as long as the dependency exists.
///
/// Binding is never null in the dependency lists of a command or in a barrier
/// wait list. It is null in the lists a scheduler-bypass submission stores on
/// its event: those dependencies are in the backend already and the list only
/// serves event::get_wait_list and dependency cleanup, so capturing the signal
/// there would keep it alive for no reason (and force a new backend event on
/// the next enqueue_signal_event, see event_impl::getHandleReusable).
struct captured_dependency {
  std::shared_ptr<event_binding> Binding;
  std::shared_ptr<event_impl> Event;
};

/// The state of one signal of an event.
///
/// An event may be enqueued for signaling more than once (see
/// sycl_ext_oneapi_reusable_events). Each such enqueue is a separate piece of
/// work with a backend event, a completion state and dependencies of its own,
/// so the state which belongs to one signal rather than to the event is kept
/// here. An event_impl points at its current binding, the command which
/// produces the signal owns the binding it writes to, and both are the same
/// object as long as the event is not enqueued for signaling again.
///
/// The binding owns the backend event handle: the last owner of the binding
/// releases it.
class event_binding {
public:
  event_binding() = default;
  event_binding(const event_binding &) = delete;
  event_binding &operator=(const event_binding &) = delete;
  ~event_binding();

  ur_event_handle_t getHandle() const { return MHandle.load(); }

  /// Sets the backend event handle. Wakes any thread waiting in
  /// event_impl::waitInternal or wait() that entered before a handle was
  /// available.
  void setHandle(ur_event_handle_t Handle) {
    MHandle.store(Handle);
    if (Handle != nullptr) {
      std::lock_guard<std::mutex> Lock(MMutex);
      MCv.notify_all();
    }
  }

  void setStateIncomplete() { MState = HES_NotComplete; }

  /// Marks the signal complete. Only for signals without a backend event; a
  /// backend event completes on its own.
  void setComplete() {
    {
      std::unique_lock<std::mutex> Lock(MMutex);
#ifndef NDEBUG
      int Expected = HES_NotComplete;
      int Desired = HES_Complete;
      bool Succeeded = MState.compare_exchange_strong(Expected, Desired);
      assert(Succeeded && "Unexpected state of event");
#else
      MState.store(static_cast<int>(HES_Complete));
#endif
    }
    MCv.notify_all();
  }

  void setEnqueued() { MIsEnqueued = true; }

  void setWorkerQueue(std::weak_ptr<queue_impl> WorkerQueue) {
    MWorkerQueue = std::move(WorkerQueue);
  }
  void setSubmittedQueue(queue_impl *SubmittedQueue);

  void setPotentiallyNativeRecorded(bool Value) {
    MPotentiallyNativeRecorded = Value;
  }

  void setSyncPoint(ur_exp_command_buffer_sync_point_t SyncPoint) {
    MSyncPoint = SyncPoint;
  }
  void setCommandBufferCommand(ur_exp_command_buffer_command_handle_t Command) {
    MCommandBufferCommand = Command;
  }

  /// Prepares the binding for another signal of the same event, when nothing
  /// but the event refers to the previous one. Keeps the backend event and the
  /// adapter; everything else is set again by the submission which follows.
  void resetForReuse() {
    assert(!MCommand && "reusing the binding of a pending command");
    MIsEnqueued = false;
    MIsFlushed = false;
    clearDependencies();
    MSubmitTime = 0;
    MHostProfilingInfo.reset();
    MSyncPoint = 0;
    MCommandBufferCommand = nullptr;
  }

  /// Waits for this signal: for the backend event if there is one, otherwise
  /// until the signal is marked complete. If the producing command has not
  /// been enqueued yet, sleeps until it is.
  void wait();

  /// Performs a flush on the queue of this signal if the user queue is
  /// different and the work producing the signal hasn't been submitted to the
  /// device yet.
  void flushIfNeeded(queue_impl *UserQueue);

  /// Drops the dependencies of this signal.
  void clearDependencies();

  /// Drops the dependencies of this signal's dependencies.
  void cleanDependenciesThroughOneLevel();

  /// Same, without locking MMutex.
  void cleanDependenciesThroughOneLevelUnlocked();

  /// The command producing this signal, or nullptr if there is none or it has
  /// been cleaned up. The scheduler graph lock must be held in read mode to
  /// read it and in write mode to set it (see event_impl::getCommand).
  Command *MCommand = nullptr;

  /// The backend event of this signal, or nullptr if the signal has no
  /// backend event (yet).
  std::atomic<ur_event_handle_t> MHandle = nullptr;
  /// The adapter MHandle belongs to; set when the event is bound to a context.
  adapter_impl *MAdapter = nullptr;

  /// Completion state. Employed only for host events and events with no
  /// backend representation (e.g. alloca). Values are HostEventState.
  std::atomic<int> MState{HES_NotComplete};

  /// Whether the work producing this signal passed enqueue.
  std::atomic<bool> MIsEnqueued{false};

  /// Whether the work producing this signal has been submitted by the queue to
  /// the device.
  std::atomic<bool> MIsFlushed{false};

  /// Guards MCv and the dependency lists below.
  std::mutex MMutex;
  /// Notified when MHandle appears or MState becomes complete.
  std::condition_variable MCv;

  /// The queue the signal belongs to.
  std::weak_ptr<queue_impl> MQueue;
  /// The queue which performs the work producing this signal.
  std::weak_ptr<queue_impl> MWorkerQueue;
  /// The queue the work was submitted to (host tasks).
  std::weak_ptr<queue_impl> MSubmittedQueue;

  /// Set from the context of the worker queue when the event is created for a
  /// command submission, marking it as potentially captured if a native graph
  /// recording was active. Used to preserve in-order dependencies that cross
  /// the native-recording capture boundary.
  bool MPotentiallyNativeRecorded = false;

  /// Submission time of the work producing this signal.
  uint64_t MSubmitTime = 0;
  /// Host-side profiling data (host tasks).
  std::unique_ptr<HostProfilingInfo> MHostProfilingInfo;

  /// If this signal is a submission to a command buffer, its sync point.
  ur_exp_command_buffer_sync_point_t MSyncPoint = 0;
  /// If this signal is a submission to a command buffer, the command-buffer
  /// command (if any) associated with it.
  ur_exp_command_buffer_command_handle_t MCommandBufferCommand = nullptr;

  /// Dependencies of the work producing this signal, prepared for waiting by
  /// the backend, and those waited for on the host. Captured when the work is
  /// submitted. See Command::processDepEvent for details.
  std::vector<captured_dependency> MPreparedDepsEvents;
  std::vector<captured_dependency> MPreparedHostDepsEvents;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
