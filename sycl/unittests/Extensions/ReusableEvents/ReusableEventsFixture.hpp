//==-- ReusableEventsFixture.hpp --- Mock backend for reusable events -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A mock backend shared by the reusable events tests. It gives every backend
// event a distinct fake handle and records what reaches the backend: the wait
// lists, the events each operation signals, retains and releases, flushes,
// finishes and frees.
//
// Every backend event is complete unless a test asks otherwise: a test may
// mark handles pending (markPending, or KernelsStayPending/BarriersStayPending
// for the events the mock produces) and complete them later. urEventWait and
// urQueueFinish block on pending handles only, and give up after a deadline,
// so a missing completion fails the test instead of hanging it.
//
// The host tasks run on the runtime's thread pool, which is created once per
// process; ReusableEventsFixture.cpp sizes it for the tests which block more
// than one host task at a time.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <gtest/gtest.h>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

#include <sycl/ext/oneapi/experimental/ipc_event.hpp>
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>

#include <detail/context_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/queue_impl.hpp>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

class BindingTestKernel;
MOCK_INTEGRATION_HEADER(BindingTestKernel)

namespace reusable_events_test {

namespace syclex = sycl::ext::oneapi::experimental;

// How long a blocking mock waits for a pending handle before it gives up.
inline constexpr std::chrono::seconds BackendDeadline{10};

// Every backend event the mock creates gets a distinct fake handle, so that
// the tests can tell the signals apart.
inline std::uintptr_t NextEventHandle = 0x1000;
// The backend events created by urEventCreateExp, i.e. for signals, with the
// flags they were created with.
inline std::vector<ur_event_handle_t> CreatedEvents;
inline std::vector<ur_exp_event_flags_t> CreatedEventFlags;
// Whether each of them was asked for low-power waits.
inline std::vector<bool> CreatedEventLowPower;
// Every handle the mock handed out, for telling foreign handles apart.
inline std::set<ur_event_handle_t> KnownHandles;
inline std::map<ur_event_handle_t, int> RetainCounts;
inline std::map<ur_event_handle_t, int> ReleaseCounts;

// What reached the backend.
inline std::vector<std::vector<ur_event_handle_t>> KernelLaunchWaitLists;
inline std::vector<ur_event_handle_t> KernelEvents;
inline std::vector<std::vector<ur_event_handle_t>> BarrierWaitLists;
// The event each barrier call was asked to signal: the handle passed in, or
// nullptr if the backend was asked to create one.
inline std::vector<ur_event_handle_t> BarrierOutEvents;
inline std::vector<std::vector<ur_event_handle_t>> CommandBufferWaitLists;
inline std::vector<ur_event_handle_t> CommandBufferEvents;
// USM copies, fills and prefetches.
inline std::vector<std::vector<ur_event_handle_t>> MemoryWaitLists;
inline std::vector<ur_event_handle_t> MemoryEvents;
inline std::vector<ur_event_handle_t> WaitedEvents;
inline std::vector<ur_queue_handle_t> FlushedQueues;
inline std::vector<ur_queue_handle_t> FinishedQueues;
inline std::vector<void *> FreedPointers;
// Wait lists which had a null entry; the backend does not accept those.
inline std::vector<std::vector<ur_event_handle_t>> NullEntryWaitLists;

// Completion control. A pending handle reports UR_EVENT_STATUS_SUBMITTED and
// blocks urEventWait (and urQueueFinish on the queue it was enqueued to) until
// it is completed.
inline std::set<ur_event_handle_t> PendingEvents;
inline std::map<ur_event_handle_t, ur_queue_handle_t> HandleQueues;
// Whether the events of kernel launches and command buffers, and those of
// barriers (signals included), start pending.
inline bool KernelsStayPending = false;
inline bool BarriersStayPending = false;
inline bool BackendWaitTimedOut = false;

// Whether the device reports native support for reusable events (and IPC
// events, which imply it).
inline bool NativeSupport = true;

inline std::mutex BackendMutex;
inline std::condition_variable BackendCv;

// The context of the test, for the backend events imported through
// ipc::event::open (the runtime checks which context they belong to).
inline ur_context_handle_t TestContextHandle = nullptr;

// Called with BackendMutex held.
inline ur_event_handle_t newFakeEvent() {
  auto Handle = reinterpret_cast<ur_event_handle_t>(NextEventHandle++);
  KnownHandles.insert(Handle);
  return Handle;
}

// Called with BackendMutex held, for an event an enqueue on Queue signals.
inline void producedBy(ur_event_handle_t Handle, ur_queue_handle_t Queue,
                       bool StaysPending) {
  HandleQueues[Handle] = Queue;
  if (StaysPending)
    PendingEvents.insert(Handle);
}

// Called with BackendMutex held.
inline std::vector<ur_event_handle_t>
recordWaitList(const ur_event_handle_t *List, uint32_t Size) {
  std::vector<ur_event_handle_t> WaitList(List, List + Size);
  if (std::find(WaitList.begin(), WaitList.end(), nullptr) != WaitList.end())
    NullEntryWaitLists.push_back(WaitList);
  return WaitList;
}

// Blocks on Lock (holding BackendMutex) until Predicate holds or the deadline
// passes.
template <typename PredicateT>
void blockUntil(std::unique_lock<std::mutex> &Lock, PredicateT Predicate) {
  if (!BackendCv.wait_for(Lock, BackendDeadline, Predicate))
    BackendWaitTimedOut = true;
}

inline void setKernelsStayPending(bool Value) {
  std::lock_guard<std::mutex> Lock(BackendMutex);
  KernelsStayPending = Value;
}

inline void setBarriersStayPending(bool Value) {
  std::lock_guard<std::mutex> Lock(BackendMutex);
  BarriersStayPending = Value;
}

inline void markPending(ur_event_handle_t Handle) {
  std::lock_guard<std::mutex> Lock(BackendMutex);
  PendingEvents.insert(Handle);
}

inline void complete(ur_event_handle_t Handle) {
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    PendingEvents.erase(Handle);
  }
  BackendCv.notify_all();
}

inline void completeAll() {
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    PendingEvents.clear();
    KernelsStayPending = false;
    BarriersStayPending = false;
  }
  BackendCv.notify_all();
}

inline bool isPending(ur_event_handle_t Handle) {
  std::lock_guard<std::mutex> Lock(BackendMutex);
  return PendingEvents.count(Handle) != 0;
}

// Completes every pending backend event when the scope is left, so that a
// failed assertion does not leave a queue destructor waiting for one.
struct CompleteAllAtScopeExit {
  ~CompleteAllAtScopeExit() { completeAll(); }
};

inline ur_result_t redefinedUrEventCreateExp(void *pParams) {
  auto params = *static_cast<ur_event_create_exp_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(BackendMutex);
  ur_event_handle_t Handle = newFakeEvent();
  **params.pphEvent = Handle;
  CreatedEvents.push_back(Handle);
  const ur_exp_event_desc_t *Desc = *params.ppEventDesc;
  CreatedEventFlags.push_back(Desc ? Desc->flags : 0);
  bool LowPower = false;
  if (Desc && Desc->pNext) {
    auto *SyncMode =
        static_cast<const ur_exp_event_sync_mode_desc_t *>(Desc->pNext);
    LowPower = SyncMode->stype == UR_STRUCTURE_TYPE_EXP_EVENT_SYNC_MODE_DESC &&
               (SyncMode->flags & UR_EXP_EVENT_SYNC_MODE_FLAG_LOW_POWER_WAIT);
  }
  CreatedEventLowPower.push_back(LowPower);
  return UR_RESULT_SUCCESS;
}

inline ur_result_t redefinedUrEventRetain(void *pParams) {
  auto params = *static_cast<ur_event_retain_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(BackendMutex);
  RetainCounts[*params.phEvent]++;
  return UR_RESULT_SUCCESS;
}

inline ur_result_t redefinedUrEventRelease(void *pParams) {
  auto params = *static_cast<ur_event_release_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(BackendMutex);
  ReleaseCounts[*params.phEvent]++;
  return UR_RESULT_SUCCESS;
}

inline ur_result_t redefinedUrEventGetInfo(void *pParams) {
  auto params = *static_cast<ur_event_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_EVENT_INFO_COMMAND_EXECUTION_STATUS) {
    auto *Result = reinterpret_cast<ur_event_status_t *>(*params.ppPropValue);
    *Result = isPending(*params.phEvent) ? UR_EVENT_STATUS_SUBMITTED
                                         : UR_EVENT_STATUS_COMPLETE;
  } else if (*params.ppropName == UR_EVENT_INFO_CONTEXT) {
    if (*params.ppPropValue)
      *static_cast<ur_context_handle_t *>(*params.ppPropValue) =
          TestContextHandle;
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(ur_context_handle_t);
  }
  return UR_RESULT_SUCCESS;
}

// An imported IPC event gets a distinct fake handle too.
inline ur_result_t redefinedUrIPCOpenEventHandleExp(void *pParams) {
  auto params = *static_cast<ur_ipc_open_event_handle_exp_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(BackendMutex);
  **params.pphEvent = newFakeEvent();
  return UR_RESULT_SUCCESS;
}

inline ur_result_t redefinedUrEventWait(void *pParams) {
  auto params = *static_cast<ur_event_wait_params_t *>(pParams);
  std::unique_lock<std::mutex> Lock(BackendMutex);
  std::vector<ur_event_handle_t> Events =
      recordWaitList(*params.pphEventWaitList, *params.pnumEvents);
  WaitedEvents.insert(WaitedEvents.end(), Events.begin(), Events.end());
  BackendCv.notify_all();
  blockUntil(Lock, [&] {
    return std::none_of(Events.begin(), Events.end(), [](ur_event_handle_t E) {
      return PendingEvents.count(E) != 0;
    });
  });
  return UR_RESULT_SUCCESS;
}

inline ur_result_t redefinedUrQueueFinish(void *pParams) {
  auto params = *static_cast<ur_queue_finish_params_t *>(pParams);
  const ur_queue_handle_t Queue = *params.phQueue;
  std::unique_lock<std::mutex> Lock(BackendMutex);
  FinishedQueues.push_back(Queue);
  BackendCv.notify_all();
  blockUntil(Lock, [&] {
    return std::none_of(
        PendingEvents.begin(), PendingEvents.end(), [&](ur_event_handle_t E) {
          auto It = HandleQueues.find(E);
          return It != HandleQueues.end() && It->second == Queue;
        });
  });
  return UR_RESULT_SUCCESS;
}

inline ur_result_t redefinedUrQueueFlush(void *pParams) {
  auto params = *static_cast<ur_queue_flush_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(BackendMutex);
  FlushedQueues.push_back(*params.phQueue);
  return UR_RESULT_SUCCESS;
}

inline ur_result_t after_urUSMFree(void *pParams) {
  auto params = *static_cast<ur_usm_free_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(BackendMutex);
  FreedPointers.push_back(*params.ppMem);
  return UR_RESULT_SUCCESS;
}

inline ur_result_t redefinedUrEnqueueKernelLaunchWithArgsExp(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_kernel_launch_with_args_exp_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(BackendMutex);
  KernelLaunchWaitLists.push_back(
      recordWaitList(*params.pphEventWaitList, *params.pnumEventsInWaitList));
  ur_event_handle_t Handle = newFakeEvent();
  producedBy(Handle, *params.phQueue, KernelsStayPending);
  KernelEvents.push_back(Handle);
  if (*params.pphEvent)
    **params.pphEvent = Handle;
  return UR_RESULT_SUCCESS;
}

inline ur_result_t redefinedUrEnqueueCommandBufferExp(void *pParams) {
  auto params = *static_cast<ur_enqueue_command_buffer_exp_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(BackendMutex);
  CommandBufferWaitLists.push_back(
      recordWaitList(*params.pphEventWaitList, *params.pnumEventsInWaitList));
  ur_event_handle_t Handle = newFakeEvent();
  producedBy(Handle, *params.phQueue, KernelsStayPending);
  CommandBufferEvents.push_back(Handle);
  if (*params.pphEvent)
    **params.pphEvent = Handle;
  return UR_RESULT_SUCCESS;
}

// Called with BackendMutex held.
inline void recordMemoryOp(ur_queue_handle_t Queue,
                           const ur_event_handle_t *List, uint32_t Size,
                           ur_event_handle_t *OutEvent) {
  MemoryWaitLists.push_back(recordWaitList(List, Size));
  ur_event_handle_t Handle = newFakeEvent();
  producedBy(Handle, Queue, KernelsStayPending);
  MemoryEvents.push_back(Handle);
  if (OutEvent)
    *OutEvent = Handle;
}

inline ur_result_t redefinedUrEnqueueUSMMemcpy(void *pParams) {
  auto params = *static_cast<ur_enqueue_usm_memcpy_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(BackendMutex);
  recordMemoryOp(*params.phQueue, *params.pphEventWaitList,
                 *params.pnumEventsInWaitList, *params.pphEvent);
  return UR_RESULT_SUCCESS;
}

inline ur_result_t redefinedUrEnqueueUSMFill(void *pParams) {
  auto params = *static_cast<ur_enqueue_usm_fill_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(BackendMutex);
  recordMemoryOp(*params.phQueue, *params.pphEventWaitList,
                 *params.pnumEventsInWaitList, *params.pphEvent);
  return UR_RESULT_SUCCESS;
}

inline ur_result_t redefinedUrEnqueueUSMPrefetch(void *pParams) {
  auto params = *static_cast<ur_enqueue_usm_prefetch_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(BackendMutex);
  recordMemoryOp(*params.phQueue, *params.pphEventWaitList,
                 *params.pnumEventsInWaitList, *params.pphEvent);
  return UR_RESULT_SUCCESS;
}

// A barrier enqueued by the scheduler may express its dependencies through a
// plain events wait preceding the barrier call; record those too.
inline ur_result_t redefinedUrEnqueueEventsWait(void *pParams) {
  auto params = *static_cast<ur_enqueue_events_wait_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(BackendMutex);
  BarrierWaitLists.push_back(
      recordWaitList(*params.pphEventWaitList, *params.pnumEventsInWaitList));
  if (*params.pphEvent) {
    if (**params.pphEvent == nullptr)
      **params.pphEvent = newFakeEvent();
    producedBy(**params.pphEvent, *params.phQueue, BarriersStayPending);
  }
  return UR_RESULT_SUCCESS;
}

inline ur_result_t redefinedUrEnqueueEventsWaitWithBarrierExt(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_events_wait_with_barrier_ext_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(BackendMutex);
  BarrierWaitLists.push_back(
      recordWaitList(*params.pphEventWaitList, *params.pnumEventsInWaitList));
  // A signal passes the reusable event in; any other barrier asks for a new
  // event.
  BarrierOutEvents.push_back(*params.pphEvent ? **params.pphEvent : nullptr);
  if (*params.pphEvent) {
    if (**params.pphEvent == nullptr)
      **params.pphEvent = newFakeEvent();
    producedBy(**params.pphEvent, *params.phQueue, BarriersStayPending);
  }
  return UR_RESULT_SUCCESS;
}

inline ur_result_t after_urDeviceGetInfo(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_DEVICE_INFO_REUSABLE_EVENTS_SUPPORT_EXP ||
      *params.ppropName == UR_DEVICE_INFO_IPC_EVENT_SUPPORT_EXP ||
      *params.ppropName == UR_DEVICE_INFO_PER_EVENT_PROFILING_SUPPORT_EXP) {
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(ur_bool_t);
    if (*params.ppPropValue)
      *static_cast<ur_bool_t *>(*params.ppPropValue) = ur_bool_t{NativeSupport};
  }
  return UR_RESULT_SUCCESS;
}

// A gate for a host task to block on, so that everything submitted to an
// in-order queue after it stays inside the runtime until the gate is opened.
class HostTaskGate {
public:
  void wait() {
    std::unique_lock<std::mutex> Lock(MMutex);
    MEntered = true;
    MCv.notify_all();
    MCv.wait(Lock, [this] { return MReady; });
  }

  void open() {
    {
      std::lock_guard<std::mutex> Lock(MMutex);
      MReady = true;
    }
    MCv.notify_all();
  }

  // Waits until a host task is blocked on the gate (or a few seconds passed).
  bool waitEntered() {
    std::unique_lock<std::mutex> Lock(MMutex);
    return MCv.wait_for(Lock, BackendDeadline, [this] { return MEntered; });
  }

private:
  std::mutex MMutex;
  std::condition_variable MCv;
  bool MReady = false;
  bool MEntered = false;
};

// Opens the gate when the scope is left, so that a failed assertion does not
// leave the host task - and the queue destructor waiting for it - blocked.
struct OpenAtScopeExit {
  std::shared_ptr<HostTaskGate> Gate;
  ~OpenAtScopeExit() { Gate->open(); }
};

// The host task shares ownership of the gate, so it stays valid for as long as
// the task may look at it.
inline sycl::event blockQueue(sycl::queue &Q,
                              std::shared_ptr<HostTaskGate> Gate) {
  return Q.submit(
      [&Gate](sycl::handler &CGH) { CGH.host_task([Gate] { Gate->wait(); }); });
}

// The barrier (or the events wait preceding it) calls which had something to
// wait for, i.e. not the signals.
inline std::vector<std::vector<ur_event_handle_t>> barriersWithWaitList() {
  std::lock_guard<std::mutex> Lock(BackendMutex);
  std::vector<std::vector<ur_event_handle_t>> Result;
  for (const auto &WaitList : BarrierWaitLists)
    if (!WaitList.empty())
      Result.push_back(WaitList);
  return Result;
}

inline int releases(ur_event_handle_t Handle) {
  std::lock_guard<std::mutex> Lock(BackendMutex);
  return ReleaseCounts[Handle];
}

inline int retains(ur_event_handle_t Handle) {
  std::lock_guard<std::mutex> Lock(BackendMutex);
  return RetainCounts[Handle];
}

// Whether every reference to a handle the mock created has been given back:
// the creation and each retain are matched by a release. Called with
// BackendMutex held, e.g. from an eventually predicate.
inline bool ownershipBalancedLocked(ur_event_handle_t Handle) {
  return 1 + RetainCounts[Handle] == ReleaseCounts[Handle];
}

inline bool ownershipBalanced(ur_event_handle_t Handle) {
  std::lock_guard<std::mutex> Lock(BackendMutex);
  return ownershipBalancedLocked(Handle);
}

inline bool waitedFor(ur_event_handle_t Handle) {
  std::lock_guard<std::mutex> Lock(BackendMutex);
  return std::find(WaitedEvents.begin(), WaitedEvents.end(), Handle) !=
         WaitedEvents.end();
}

inline size_t kernelLaunches() {
  std::lock_guard<std::mutex> Lock(BackendMutex);
  return KernelLaunchWaitLists.size();
}

inline size_t flushesOf(ur_queue_handle_t Queue) {
  std::lock_guard<std::mutex> Lock(BackendMutex);
  return std::count(FlushedQueues.begin(), FlushedQueues.end(), Queue);
}

// Waits until \p Predicate holds, or a few seconds passed. The predicate runs
// with BackendMutex held.
template <typename PredicateT> bool eventually(PredicateT Predicate) {
  const auto Deadline = std::chrono::steady_clock::now() + BackendDeadline;
  while (std::chrono::steady_clock::now() < Deadline) {
    {
      std::lock_guard<std::mutex> Lock(BackendMutex);
      if (Predicate())
        return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

// A best-effort check that an operation running on \p Future is blocked: it
// has not finished within a short interval. An operation which blocks before
// it reaches the backend - waiting for a signal held in the runtime to get its
// backend event - cannot be observed entering the wait, so a pass does not
// prove the operation is waiting; a failure does prove it did not wait.
template <typename T> bool stillBlocked(std::future<T> &Future) {
  return Future.wait_for(std::chrono::milliseconds(100)) ==
         std::future_status::timeout;
}

// Whether the operation running on \p Future finished within the deadline.
template <typename T> bool finishes(std::future<T> &Future) {
  return Future.wait_for(BackendDeadline) == std::future_status::ready;
}

inline ur_event_handle_t handleOf(const sycl::event &E) {
  return sycl::detail::getSyclObjImpl(E)->getHandle();
}

inline std::shared_ptr<sycl::detail::event_binding>
bindingOf(const sycl::event &E) {
  return sycl::detail::getSyclObjImpl(E)->getBinding();
}

inline ur_queue_handle_t handleOf(const sycl::queue &Q) {
  return sycl::detail::getSyclObjImpl(Q)->getHandleRef();
}

inline bool isComplete(const sycl::event &E) {
  return E.get_info<sycl::info::event::command_execution_status>() ==
         sycl::info::event_command_status::complete;
}

class ReusableEventsTest : public ::testing::Test {
protected:
  void SetUp() override {
    NextEventHandle = 0x1000;
    CreatedEvents.clear();
    CreatedEventFlags.clear();
    CreatedEventLowPower.clear();
    KnownHandles.clear();
    RetainCounts.clear();
    ReleaseCounts.clear();
    KernelLaunchWaitLists.clear();
    KernelEvents.clear();
    BarrierWaitLists.clear();
    BarrierOutEvents.clear();
    CommandBufferWaitLists.clear();
    CommandBufferEvents.clear();
    MemoryWaitLists.clear();
    MemoryEvents.clear();
    WaitedEvents.clear();
    FlushedQueues.clear();
    FinishedQueues.clear();
    FreedPointers.clear();
    NullEntryWaitLists.clear();
    PendingEvents.clear();
    HandleQueues.clear();
    KernelsStayPending = false;
    BarriersStayPending = false;
    BackendWaitTimedOut = false;
    NativeSupport = nativeSupport();

    mock::getCallbacks().set_replace_callback("urEventCreateExp",
                                              &redefinedUrEventCreateExp);
    mock::getCallbacks().set_replace_callback("urEventRetain",
                                              &redefinedUrEventRetain);
    mock::getCallbacks().set_replace_callback("urEventRelease",
                                              &redefinedUrEventRelease);
    mock::getCallbacks().set_replace_callback("urEventGetInfo",
                                              &redefinedUrEventGetInfo);
    mock::getCallbacks().set_replace_callback("urEventWait",
                                              &redefinedUrEventWait);
    mock::getCallbacks().set_replace_callback("urQueueFinish",
                                              &redefinedUrQueueFinish);
    mock::getCallbacks().set_replace_callback("urQueueFlush",
                                              &redefinedUrQueueFlush);
    mock::getCallbacks().set_after_callback("urUSMFree", &after_urUSMFree);
    mock::getCallbacks().set_replace_callback(
        "urEnqueueKernelLaunchWithArgsExp",
        &redefinedUrEnqueueKernelLaunchWithArgsExp);
    mock::getCallbacks().set_replace_callback("urEnqueueEventsWait",
                                              &redefinedUrEnqueueEventsWait);
    mock::getCallbacks().set_replace_callback(
        "urEnqueueCommandBufferExp", &redefinedUrEnqueueCommandBufferExp);
    mock::getCallbacks().set_replace_callback("urEnqueueUSMMemcpy",
                                              &redefinedUrEnqueueUSMMemcpy);
    mock::getCallbacks().set_replace_callback("urEnqueueUSMFill",
                                              &redefinedUrEnqueueUSMFill);
    mock::getCallbacks().set_replace_callback("urEnqueueUSMPrefetch",
                                              &redefinedUrEnqueueUSMPrefetch);
    mock::getCallbacks().set_replace_callback(
        "urEnqueueEventsWaitWithBarrierExt",
        &redefinedUrEnqueueEventsWaitWithBarrierExt);
    mock::getCallbacks().set_replace_callback(
        "urIPCOpenEventHandleExp", &redefinedUrIPCOpenEventHandleExp);
    mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                            &after_urDeviceGetInfo);

    Dev = sycl::platform().get_devices()[0];
    Ctx = sycl::context{Dev};
    TestContextHandle = sycl::detail::getSyclObjImpl(Ctx)->getHandleRef();
  }

  void TearDown() override {
    completeAll();
    std::lock_guard<std::mutex> Lock(BackendMutex);
    EXPECT_FALSE(BackendWaitTimedOut)
        << "a backend wait gave up on a pending event";
    EXPECT_TRUE(NullEntryWaitLists.empty())
        << "a null event reached a backend wait list";
  }

  // Whether the device reports native reusable-event support; tests
  // parameterized over it override this.
  virtual bool nativeSupport() const { return true; }

  sycl::queue inOrderQueue() {
    return sycl::queue{Ctx, Dev, sycl::property::queue::in_order{}};
  }

  sycl::queue inOrderQueue(const sycl::context &C) {
    return sycl::queue{C, Dev, sycl::property::queue::in_order{}};
  }

  sycl::unittest::UrMock<> Mock;
  sycl::device Dev;
  sycl::context Ctx;
};

// The same, with and without native reusable-event support.
class ReusableEventsSupportTest : public ReusableEventsTest,
                                  public ::testing::WithParamInterface<bool> {
protected:
  bool nativeSupport() const override { return GetParam(); }
};

inline std::string supportName(const ::testing::TestParamInfo<bool> &Info) {
  return Info.param ? "NativeSupport" : "NoNativeSupport";
}

} // namespace reusable_events_test
