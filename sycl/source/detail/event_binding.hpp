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
class event_impl;
class queue_impl;

/// Completion state of a signal without a backend event (host events, alloca
/// and the like).
enum HostEventState : int { HES_NotComplete = 0, HES_Complete, HES_Discarded };

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
  /// event_impl::waitInternal that entered before a handle was available.
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

  /// Dependency events prepared for waiting by backend.
  /// See Command::processDepEvent for details.
  std::vector<std::shared_ptr<event_impl>> MPreparedDepsEvents;
  std::vector<std::shared_ptr<event_impl>> MPreparedHostDepsEvents;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
