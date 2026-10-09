//==-- ReusableEventsScheduler.cpp --- Signals inside the scheduler -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A command held in the scheduler captures the signals it depends on; the
// event may be signaled again before the command reaches the backend. The
// tests below hold such commands - barriers, memory commands, cross-context
// bridges, stream flushes, host fallbacks, buffer cleanup, auxiliary resources
// and failed enqueues - re-signal the event, and check that the command still
// follows, waits for and reports the signal it captured.
//
// On top of the shared mock, the tests record every wait list a queue
// receives together with the queue, so that a handle from another context
// can be told apart; they record the buffer commands of the memory manager
// and the buffer allocations and releases; and they inject failures into the
// creation of backend events, barriers and waits.
//
// The scenario numbers refer to the reusable events test plan
// (tests-10-02.md); the findings to review-10-01.md and review-10-02.md.
//
//===----------------------------------------------------------------------===//

#include "ReusableEventsFixture.hpp"

#include <detail/accessor_impl.hpp>
#include <detail/scheduler/scheduler.hpp>

#include <atomic>
#include <cstring>
#include <functional>
#include <numeric>
#include <optional>
#include <string>

namespace {

using namespace reusable_events_test;

// A wait list a queue received: kernel launches, barriers, events waits and
// buffer commands.
struct QueueCall {
  ur_queue_handle_t Queue;
  std::vector<ur_event_handle_t> WaitList;
};

// A buffer command of the memory manager.
struct MemoryCall {
  std::string Name;
  ur_queue_handle_t Queue;
  std::vector<ur_event_handle_t> WaitList;
};

enum class FailurePoint { None, EventCreate, EventsWait, Barrier };

// All of the below are guarded by BackendMutex.
std::vector<QueueCall> QueueCalls;
std::map<ur_queue_handle_t, ur_context_handle_t> QueueContexts;
std::vector<MemoryCall> MemoryCalls;
// The buffer commands and buffer releases, in order.
std::vector<std::string> MemoryLog;
int MemCreates = 0;
int MemRetains = 0;
int MemReleases = 0;
// Whether the events of buffer commands start pending, and those which did.
bool MemoryOpsStayPending = false;
std::vector<ur_event_handle_t> HeldMemoryEvents;
// The next failure to inject, and on which queue (the event creation has no
// queue).
FailurePoint ArmedFailure = FailurePoint::None;
ur_queue_handle_t FailureQueue = nullptr;
int InjectedFailures = 0;
// The handle a failed barrier was asked to signal.
ur_event_handle_t FailedBarrierEvent = nullptr;
// Every urEventWait on this handle fails.
ur_event_handle_t FailingWait = nullptr;

// The resources a test registers with the scheduler count their deletion
// here; they may outlive a failed test, so this is not a local.
std::atomic<int> DeletedResources{0};

template <typename C, typename V>
bool contains(const C &Container, const V &V0) {
  return std::find(Container.begin(), Container.end(), V0) != Container.end();
}

// Called with BackendMutex held.
bool takeFailure(FailurePoint Point, ur_queue_handle_t Queue) {
  if (ArmedFailure != Point || (FailureQueue && FailureQueue != Queue))
    return false;
  ArmedFailure = FailurePoint::None;
  ++InjectedFailures;
  return true;
}

void armFailure(FailurePoint Point, ur_queue_handle_t Queue) {
  std::lock_guard<std::mutex> Lock(BackendMutex);
  ArmedFailure = Point;
  FailureQueue = Queue;
}

int injectedFailures() {
  std::lock_guard<std::mutex> Lock(BackendMutex);
  return InjectedFailures;
}

void setFailingWait(ur_event_handle_t Handle) {
  std::lock_guard<std::mutex> Lock(BackendMutex);
  FailingWait = Handle;
}

void setMemoryOpsStayPending(bool Value) {
  std::lock_guard<std::mutex> Lock(BackendMutex);
  MemoryOpsStayPending = Value;
}

void completeHeldMemoryOps() {
  std::vector<ur_event_handle_t> Held;
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    Held = HeldMemoryEvents;
  }
  for (ur_event_handle_t Handle : Held)
    complete(Handle);
}

ur_result_t beforeKernelLaunch(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_kernel_launch_with_args_exp_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(BackendMutex);
  const ur_event_handle_t *List = *params.pphEventWaitList;
  QueueCalls.push_back(
      {*params.phQueue, {List, List + *params.pnumEventsInWaitList}});
  return UR_RESULT_SUCCESS;
}

ur_result_t beforeEventsWait(void *pParams) {
  auto params = *static_cast<ur_enqueue_events_wait_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(BackendMutex);
  const ur_event_handle_t *List = *params.pphEventWaitList;
  QueueCalls.push_back(
      {*params.phQueue, {List, List + *params.pnumEventsInWaitList}});
  if (takeFailure(FailurePoint::EventsWait, *params.phQueue))
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  return UR_RESULT_SUCCESS;
}

ur_result_t beforeBarrier(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_events_wait_with_barrier_ext_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(BackendMutex);
  const ur_event_handle_t *List = *params.pphEventWaitList;
  QueueCalls.push_back(
      {*params.phQueue, {List, List + *params.pnumEventsInWaitList}});
  if (!takeFailure(FailurePoint::Barrier, *params.phQueue))
    return UR_RESULT_SUCCESS;
  // The handle passed in is never signaled: a waiter for it blocks until the
  // test completes it.
  if (*params.pphEvent && **params.pphEvent) {
    FailedBarrierEvent = **params.pphEvent;
    PendingEvents.insert(FailedBarrierEvent);
  }
  return UR_RESULT_ERROR_OUT_OF_RESOURCES;
}

ur_result_t beforeEventCreate(void *) {
  std::lock_guard<std::mutex> Lock(BackendMutex);
  if (takeFailure(FailurePoint::EventCreate, nullptr))
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  return UR_RESULT_SUCCESS;
}

ur_result_t beforeEventWait(void *pParams) {
  auto params = *static_cast<ur_event_wait_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(BackendMutex);
  if (!FailingWait)
    return UR_RESULT_SUCCESS;
  const ur_event_handle_t *List = *params.pphEventWaitList;
  if (std::find(List, List + *params.pnumEvents, FailingWait) ==
      List + *params.pnumEvents)
    return UR_RESULT_SUCCESS;
  ++InjectedFailures;
  return UR_RESULT_ERROR_OUT_OF_RESOURCES;
}

// The default mock implementation of a buffer command gives its event a
// dummy handle; this runs after it.
template <typename ParamsT>
ur_result_t afterMemoryCall(const char *Name, void *pParams) {
  auto params = *static_cast<ParamsT *>(pParams);
  std::lock_guard<std::mutex> Lock(BackendMutex);
  std::vector<ur_event_handle_t> WaitList =
      recordWaitList(*params.pphEventWaitList, *params.pnumEventsInWaitList);
  QueueCalls.push_back({*params.phQueue, WaitList});
  MemoryCalls.push_back({Name, *params.phQueue, WaitList});
  MemoryLog.push_back(Name);
  if (MemoryOpsStayPending && *params.pphEvent && **params.pphEvent) {
    producedBy(**params.pphEvent, *params.phQueue, /*StaysPending*/ true);
    HeldMemoryEvents.push_back(**params.pphEvent);
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t afterMemCreate(void *) {
  std::lock_guard<std::mutex> Lock(BackendMutex);
  ++MemCreates;
  return UR_RESULT_SUCCESS;
}

ur_result_t beforeMemRetain(void *) {
  std::lock_guard<std::mutex> Lock(BackendMutex);
  ++MemRetains;
  return UR_RESULT_SUCCESS;
}

ur_result_t beforeMemRelease(void *) {
  std::lock_guard<std::mutex> Lock(BackendMutex);
  ++MemReleases;
  MemoryLog.push_back("Release");
  return UR_RESULT_SUCCESS;
}

// Every pointer is unknown to the backend, i.e. a host pointer.
ur_result_t afterUSMGetMemAllocInfo(void *pParams) {
  auto params = *static_cast<ur_usm_get_mem_alloc_info_params_t *>(pParams);
  if (*params.ppropName == UR_USM_ALLOC_INFO_TYPE && *params.ppPropValue)
    *static_cast<ur_usm_type_t *>(*params.ppPropValue) = UR_USM_TYPE_UNKNOWN;
  return UR_RESULT_SUCCESS;
}

// The profiling timestamps of a handle are distinct from those of every other
// handle.
uint64_t profilingValue(ur_event_handle_t Handle, ur_profiling_info_t Info) {
  uint64_t Code = 0;
  switch (Info) {
  case UR_PROFILING_INFO_COMMAND_QUEUED:
    Code = 1;
    break;
  case UR_PROFILING_INFO_COMMAND_SUBMIT:
    Code = 2;
    break;
  case UR_PROFILING_INFO_COMMAND_START:
    Code = 3;
    break;
  case UR_PROFILING_INFO_COMMAND_END:
    Code = 4;
    break;
  case UR_PROFILING_INFO_COMMAND_COMPLETE:
    Code = 5;
    break;
  default:
    break;
  }
  return reinterpret_cast<std::uintptr_t>(Handle) * 16 + Code;
}

ur_result_t replaceEventGetProfilingInfo(void *pParams) {
  auto params = *static_cast<ur_event_get_profiling_info_params_t *>(pParams);
  if (*params.ppPropValue)
    *static_cast<uint64_t *>(*params.ppPropValue) =
        profilingValue(*params.phEvent, *params.ppropName);
  if (*params.ppPropSizeRet)
    **params.ppPropSizeRet = sizeof(uint64_t);
  return UR_RESULT_SUCCESS;
}

// Called with BackendMutex held.
bool inSomeWaitListLocked(ur_event_handle_t Handle) {
  return std::any_of(
      QueueCalls.begin(), QueueCalls.end(),
      [&](const QueueCall &Call) { return contains(Call.WaitList, Handle); });
}

bool inSomeWaitList(ur_event_handle_t Handle) {
  std::lock_guard<std::mutex> Lock(BackendMutex);
  return inSomeWaitListLocked(Handle);
}

// Called with BackendMutex held.
bool submittedOnLocked(ur_queue_handle_t Queue, ur_event_handle_t Handle) {
  return std::any_of(
      QueueCalls.begin(), QueueCalls.end(), [&](const QueueCall &Call) {
        return Call.Queue == Queue && contains(Call.WaitList, Handle);
      });
}

// The non-empty wait lists the queue received.
std::vector<std::vector<ur_event_handle_t>>
waitListsOn(ur_queue_handle_t Queue) {
  std::lock_guard<std::mutex> Lock(BackendMutex);
  std::vector<std::vector<ur_event_handle_t>> Result;
  for (const QueueCall &Call : QueueCalls)
    if (Call.Queue == Queue && !Call.WaitList.empty())
      Result.push_back(Call.WaitList);
  return Result;
}

// Remembers the context of a queue, for foreignHandleSubmitted.
void track(const sycl::queue &Q) {
  ur_context_handle_t Context =
      sycl::detail::getSyclObjImpl(Q.get_context())->getHandleRef();
  std::lock_guard<std::mutex> Lock(BackendMutex);
  QueueContexts[handleOf(Q)] = Context;
}

// Whether a tracked queue received a handle produced on a tracked queue of
// another context.
bool foreignHandleSubmitted() {
  std::lock_guard<std::mutex> Lock(BackendMutex);
  for (const QueueCall &Call : QueueCalls) {
    auto To = QueueContexts.find(Call.Queue);
    if (To == QueueContexts.end())
      continue;
    for (ur_event_handle_t Handle : Call.WaitList) {
      auto Producer = HandleQueues.find(Handle);
      if (Producer == HandleQueues.end())
        continue;
      auto From = QueueContexts.find(Producer->second);
      if (From != QueueContexts.end() && From->second != To->second)
        return true;
    }
  }
  return false;
}

std::vector<ur_event_handle_t> kernelWaitList(size_t Index) {
  std::lock_guard<std::mutex> Lock(BackendMutex);
  if (Index >= KernelLaunchWaitLists.size())
    return {};
  return KernelLaunchWaitLists[Index];
}

size_t createdEvents() {
  std::lock_guard<std::mutex> Lock(BackendMutex);
  return CreatedEvents.size();
}

// Signals E on Q; the backend event of the signal stays pending until the test
// completes it.
void signalPending(sycl::queue &Q, sycl::event &E) {
  setBarriersStayPending(true);
  syclex::enqueue_signal_event(Q, E);
  setBarriersStayPending(false);
}

sycl::event submitKernel(sycl::queue &Q,
                         const std::vector<sycl::event> &Deps = {}) {
  return Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Deps);
    CGH.single_task<BindingTestKernel>([] {});
  });
}

sycl::event writeKernel(sycl::queue &Q, sycl::buffer<int, 1> &Buf) {
  return Q.submit([&](sycl::handler &CGH) {
    sycl::accessor Acc{Buf, CGH, sycl::write_only};
    CGH.single_task<BindingTestKernel>([] {});
  });
}

// Waits for E; whether the wait threw does not matter to the caller.
bool waitQuietly(sycl::event E) {
  try {
    E.wait();
    return true;
  } catch (const sycl::exception &) {
    return false;
  }
}

// Counts the asynchronous errors reported to a queue.
struct AsyncErrors {
  std::shared_ptr<std::atomic<int>> Count =
      std::make_shared<std::atomic<int>>(0);

  sycl::async_handler handler() const {
    return [Count = Count](sycl::exception_list List) {
      *Count += static_cast<int>(List.size());
    };
  }

  int count() const { return *Count; }
};

// The protected members of the scheduler the tests drive directly. The struct
// is never instantiated: the member pointers are formed through it and
// applied to the scheduler.
struct SchedulerAccess : sycl::detail::Scheduler {
  static bool leavesCompleted(sycl::detail::MemObjRecord *Record) {
    sycl::detail::Scheduler &S = getInstance();
    auto Lock = (S.*&SchedulerAccess::acquireReadLock)();
    return (S.*&SchedulerAccess::checkLeavesCompletion)(Record);
  }

  static void
  registerResources(std::shared_ptr<sycl::detail::event_impl> Event,
                    std::vector<std::shared_ptr<const void>> Resources) {
    sycl::detail::Scheduler &S = getInstance();
    (S.*&SchedulerAccess::registerAuxiliaryResources)(Event,
                                                      std::move(Resources));
  }

  static void
  takeResources(const std::shared_ptr<sycl::detail::event_impl> &Dst,
                const std::shared_ptr<sycl::detail::event_impl> &Src) {
    sycl::detail::Scheduler &S = getInstance();
    (S.*&SchedulerAccess::takeAuxiliaryResources)(Dst, Src);
  }

  static void cleanupResources(sycl::detail::BlockingT Blocking) {
    sycl::detail::Scheduler &S = getInstance();
    (S.*&SchedulerAccess::cleanupAuxiliaryResources)(Blocking);
  }
};

sycl::detail::MemObjRecord *recordOf(sycl::buffer<int, 1> &Buf) {
  sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::target::device>
      Placeholder{Buf};
  return sycl::detail::Scheduler::getMemObjRecord(
      sycl::detail::getSyclObjImpl(
          static_cast<const sycl::detail::AccessorBaseHost &>(Placeholder))
          .get());
}

template <typename T>
std::string paramName(const ::testing::TestParamInfo<T> &Info) {
  return Info.param.Name;
}

enum class BarrierApi { Handler, Queue };

void submitBarrier(sycl::queue &Q, const sycl::event &E, BarrierApi Api) {
  if (Api == BarrierApi::Handler)
    Q.submit([&](sycl::handler &CGH) { CGH.ext_oneapi_barrier({E}); });
  else
    Q.ext_oneapi_submit_barrier(std::vector<sycl::event>{E});
}

class ReusableEventsSchedulerTest : public ReusableEventsTest {
protected:
  void SetUp() override {
    ReusableEventsTest::SetUp();
    {
      std::lock_guard<std::mutex> Lock(BackendMutex);
      QueueCalls.clear();
      QueueContexts.clear();
      MemoryCalls.clear();
      MemoryLog.clear();
      MemCreates = 0;
      MemRetains = 0;
      MemReleases = 0;
      MemoryOpsStayPending = false;
      HeldMemoryEvents.clear();
      ArmedFailure = FailurePoint::None;
      FailureQueue = nullptr;
      InjectedFailures = 0;
      FailedBarrierEvent = nullptr;
      FailingWait = nullptr;
    }
    DeletedResources = 0;

    auto &Callbacks = mock::getCallbacks();
    Callbacks.set_before_callback("urEnqueueKernelLaunchWithArgsExp",
                                  &beforeKernelLaunch);
    Callbacks.set_before_callback("urEnqueueEventsWait", &beforeEventsWait);
    Callbacks.set_before_callback("urEnqueueEventsWaitWithBarrierExt",
                                  &beforeBarrier);
    Callbacks.set_before_callback("urEventCreateExp", &beforeEventCreate);
    Callbacks.set_before_callback("urEventWait", &beforeEventWait);
    Callbacks.set_after_callback("urEnqueueMemBufferRead", [](void *P) {
      return afterMemoryCall<ur_enqueue_mem_buffer_read_params_t>("Read", P);
    });
    Callbacks.set_after_callback("urEnqueueMemBufferReadRect", [](void *P) {
      return afterMemoryCall<ur_enqueue_mem_buffer_read_rect_params_t>(
          "ReadRect", P);
    });
    Callbacks.set_after_callback("urEnqueueMemBufferWrite", [](void *P) {
      return afterMemoryCall<ur_enqueue_mem_buffer_write_params_t>("Write", P);
    });
    Callbacks.set_after_callback("urEnqueueMemBufferWriteRect", [](void *P) {
      return afterMemoryCall<ur_enqueue_mem_buffer_write_rect_params_t>(
          "WriteRect", P);
    });
    Callbacks.set_after_callback("urEnqueueMemBufferCopy", [](void *P) {
      return afterMemoryCall<ur_enqueue_mem_buffer_copy_params_t>("Copy", P);
    });
    Callbacks.set_after_callback("urEnqueueMemBufferCopyRect", [](void *P) {
      return afterMemoryCall<ur_enqueue_mem_buffer_copy_rect_params_t>(
          "CopyRect", P);
    });
    Callbacks.set_after_callback("urEnqueueMemBufferFill", [](void *P) {
      return afterMemoryCall<ur_enqueue_mem_buffer_fill_params_t>("Fill", P);
    });
    Callbacks.set_after_callback("urEnqueueMemBufferMap", [](void *P) {
      return afterMemoryCall<ur_enqueue_mem_buffer_map_params_t>("Map", P);
    });
    Callbacks.set_after_callback("urEnqueueMemUnmap", [](void *P) {
      return afterMemoryCall<ur_enqueue_mem_unmap_params_t>("Unmap", P);
    });
    Callbacks.set_after_callback("urMemBufferCreate", &afterMemCreate);
    Callbacks.set_after_callback("urMemBufferPartition", &afterMemCreate);
    Callbacks.set_before_callback("urMemRetain", &beforeMemRetain);
    Callbacks.set_before_callback("urMemRelease", &beforeMemRelease);
  }

  // tests-10-02 U15: a barrier on a signal held in the scheduler is issued
  // once the signal is in the backend, and waits for it.
  void checkBarrierWaitsForDeferredSignal(BarrierApi Api) {
    sycl::queue SignalQ = inOrderQueue();
    sycl::queue Q = inOrderQueue();
    const ur_queue_handle_t QHandle = handleOf(Q);
    sycl::event E = syclex::make_event(Ctx);
    auto Gate = std::make_shared<HostTaskGate>();
    std::future<void> Barrier;
    std::future<void> Drain;
    CompleteAllAtScopeExit CompleteAll;
    OpenAtScopeExit OpenGate{Gate};

    blockQueue(SignalQ, Gate);
    syclex::enqueue_signal_event(SignalQ, E);
    ASSERT_EQ(handleOf(E), nullptr);

    // The submission may block until the signal is in the backend.
    Barrier = std::async(std::launch::async, [&] { submitBarrier(Q, E, Api); });
    Barrier.wait_for(std::chrono::milliseconds(100));
    EXPECT_TRUE(waitListsOn(QHandle).empty())
        << "the barrier reached the backend before the signal it waits for";

    // The backend events stay pending, so that no wait list is filtered.
    setBarriersStayPending(true);
    Gate->open();
    ASSERT_TRUE(finishes(Barrier));
    Drain = std::async(std::launch::async, [&] { Q.wait(); });
    ur_event_handle_t Signal = nullptr;
    EXPECT_TRUE(eventually([&] {
      if (CreatedEvents.empty())
        return false;
      Signal = CreatedEvents.front();
      return submittedOnLocked(QHandle, Signal);
    }));
    EXPECT_EQ(waitListsOn(QHandle),
              (std::vector<std::vector<ur_event_handle_t>>{{Signal}}));
    completeAll();
    EXPECT_TRUE(finishes(Drain));
  }

  // tests-10-02 U15, step 3: a barrier on a published but incomplete signal
  // waits for it in the backend.
  void checkBarrierWaitsForPublishedSignal(BarrierApi Api) {
    sycl::queue SignalQ = inOrderQueue();
    sycl::queue Q = inOrderQueue();
    const ur_queue_handle_t QHandle = handleOf(Q);
    sycl::event E = syclex::make_event(Ctx);
    std::future<void> Barrier;
    CompleteAllAtScopeExit CompleteAll;

    signalPending(SignalQ, E);
    const ur_event_handle_t Signal = handleOf(E);
    ASSERT_NE(Signal, nullptr);
    Barrier = std::async(std::launch::async, [&] { submitBarrier(Q, E, Api); });
    ASSERT_TRUE(finishes(Barrier));
    EXPECT_EQ(waitListsOn(QHandle),
              (std::vector<std::vector<ur_event_handle_t>>{{Signal}}));
    EXPECT_FALSE(isComplete(E));
    complete(Signal);
    Q.wait();
  }

  // tests-10-02 U15, step 4: a barrier on a signal of another context waits
  // for it on the host; the queue never receives the foreign handle.
  void checkBarrierBridgesForeignSignal(BarrierApi Api) {
    sycl::context Other{Dev};
    sycl::queue SignalQ = inOrderQueue(Other);
    sycl::queue Q = inOrderQueue();
    track(SignalQ);
    track(Q);
    sycl::event E = syclex::make_event(Other);
    std::future<void> Barrier;
    CompleteAllAtScopeExit CompleteAll;

    signalPending(SignalQ, E);
    const ur_event_handle_t Signal = handleOf(E);
    ASSERT_NE(Signal, nullptr);
    // The submission may block on the host bridge.
    Barrier = std::async(std::launch::async, [&] { submitBarrier(Q, E, Api); });
    EXPECT_TRUE(eventually([&] { return contains(WaitedEvents, Signal); }))
        << "the foreign signal was not waited for on the host";
    complete(Signal);
    EXPECT_TRUE(finishes(Barrier));
    Q.wait();
    EXPECT_FALSE(foreignHandleSubmitted());
  }

  // tests-10-02 U15, step 5: a barrier holding a captured signal keeps
  // following it when the event is signaled again.
  void checkBarrierFollowsTheCapturedSignal(BarrierApi Api) {
    sycl::queue SignalQ = inOrderQueue();
    sycl::queue ResignalQ = inOrderQueue();
    sycl::queue Q = inOrderQueue();
    const ur_queue_handle_t QHandle = handleOf(Q);
    sycl::event E = syclex::make_event(Ctx);
    auto Gate = std::make_shared<HostTaskGate>();
    std::future<void> Barrier;
    CompleteAllAtScopeExit CompleteAll;
    OpenAtScopeExit OpenGate{Gate};

    signalPending(SignalQ, E);
    const ur_event_handle_t First = handleOf(E);
    blockQueue(Q, Gate);
    Barrier = std::async(std::launch::async, [&] { submitBarrier(Q, E, Api); });
    ASSERT_TRUE(finishes(Barrier));

    signalPending(ResignalQ, E);
    const ur_event_handle_t Second = handleOf(E);
    ASSERT_NE(First, nullptr);
    ASSERT_NE(Second, nullptr);
    ASSERT_NE(First, Second);
    complete(Second);

    Gate->open();
    EXPECT_TRUE(eventually([&] { return submittedOnLocked(QHandle, First); }));
    EXPECT_FALSE(inSomeWaitList(Second));
    complete(First);
    Q.wait();
  }

  // tests-10-02 U29: a signal fails at Point while a consumer of the previous
  // signal is held and waiters for the new one are blocked.
  void checkFailedSignal(FailurePoint Point) {
    AsyncErrors SignalErrors;
    ur_event_handle_t Old = nullptr;
    ur_event_handle_t Failed = nullptr;
    {
      sycl::queue OldQ = inOrderQueue();
      sycl::queue ConsumerQ = inOrderQueue();
      sycl::queue SignalQ{Ctx, Dev, SignalErrors.handler(),
                          sycl::property::queue::in_order{}};
      sycl::event E = syclex::make_event(Ctx);
      auto ConsumerGate = std::make_shared<HostTaskGate>();
      auto SignalGate = std::make_shared<HostTaskGate>();
      std::future<bool> EarlyWaiter;
      std::future<bool> LateWaiter;
      CompleteAllAtScopeExit CompleteAll;
      OpenAtScopeExit OpenConsumer{ConsumerGate};
      OpenAtScopeExit OpenSignal{SignalGate};

      // A consumer of a valid older signal, held behind a host task.
      signalPending(OldQ, E);
      Old = handleOf(E);
      ASSERT_NE(Old, nullptr);
      blockQueue(ConsumerQ, ConsumerGate);
      const size_t Launches = kernelLaunches();
      submitKernel(ConsumerQ, {E});

      // The new signal waits in the scheduler; a waiter starts after the
      // event moved on to it.
      blockQueue(SignalQ, SignalGate);
      syclex::enqueue_signal_event(SignalQ, E);
      EarlyWaiter =
          std::async(std::launch::async, [E] { return waitQuietly(E); });

      armFailure(Point, Point == FailurePoint::EventCreate ? nullptr
                                                           : handleOf(SignalQ));
      SignalGate->open();
      const bool Reached = eventually([] { return InjectedFailures > 0; });
      if (Point == FailurePoint::EventsWait && !Reached) {
        // Best effort: at 8337da70 a signal has no prerequisite events to
        // wait for, so the failure cannot be injected; only check that the
        // signal completes.
        armFailure(FailurePoint::None, nullptr);
      } else {
        EXPECT_TRUE(Reached) << "the failure was not injected";
      }
      // Another waiter starts after the failure.
      LateWaiter =
          std::async(std::launch::async, [E] { return waitQuietly(E); });
      EXPECT_TRUE(finishes(EarlyWaiter));
      EXPECT_TRUE(finishes(LateWaiter));
      const bool EarlyOk = EarlyWaiter.get();
      const bool LateOk = LateWaiter.get();
      SignalQ.throw_asynchronous();
      if (Reached) {
        // The failure is reported to the thread which enqueued the signal: a
        // waiter, or the host task's queue.
        EXPECT_GE(SignalErrors.count() + !EarlyOk + !LateOk, 1);
      }
      {
        std::lock_guard<std::mutex> Lock(BackendMutex);
        Failed = FailedBarrierEvent;
      }

      // The older signal and its consumer are not affected.
      ConsumerGate->open();
      EXPECT_TRUE(eventually(
          [&] { return KernelLaunchWaitLists.size() == Launches + 1; }));
      EXPECT_EQ(kernelWaitList(Launches), std::vector<ur_event_handle_t>{Old});
      complete(Old);
      ConsumerQ.wait();
      if (Failed)
        complete(Failed);
    }
    EXPECT_TRUE(eventually([&] { return ownershipBalancedLocked(Old); }));
    if (Failed) {
      EXPECT_TRUE(eventually([&] { return ownershipBalancedLocked(Failed); }))
          << "the backend event of the failed signal leaked";
    }
  }
};

class ReusableEventsSchedulerNoNativeTest : public ReusableEventsSchedulerTest {
protected:
  bool nativeSupport() const override { return false; }
};

//===----------------------------------------------------------------------===//
// U15: handler barriers.
//===----------------------------------------------------------------------===//

// tests-10-02 U15, steps 1-2.
// Known defect: review-10-02 #6 (getUrEventsBlocking takes the handle of a
// signal a deferred BLOCKING enqueue left in the scheduler, i.e. null).
TEST_F(ReusableEventsSchedulerTest,
       DISABLED_HandlerBarrierWaitsForDeferredSignal) {
  checkBarrierWaitsForDeferredSignal(BarrierApi::Handler);
}

// tests-10-02 U15, steps 1-2, queue barrier control.
TEST_F(ReusableEventsSchedulerTest, QueueBarrierWaitsForDeferredSignal) {
  checkBarrierWaitsForDeferredSignal(BarrierApi::Queue);
}

// tests-10-02 U15, step 3.
TEST_F(ReusableEventsSchedulerTest, HandlerBarrierWaitsForPublishedSignal) {
  checkBarrierWaitsForPublishedSignal(BarrierApi::Handler);
}

// tests-10-02 U15, step 3, queue barrier control.
TEST_F(ReusableEventsSchedulerTest, QueueBarrierWaitsForPublishedSignal) {
  checkBarrierWaitsForPublishedSignal(BarrierApi::Queue);
}

// tests-10-02 U15, step 4.
// Known defect: review-10-02 #6 (the handler barrier wait list passes the
// handle of another context to the queue instead of bridging it).
TEST_F(ReusableEventsSchedulerTest,
       DISABLED_HandlerBarrierBridgesForeignSignal) {
  checkBarrierBridgesForeignSignal(BarrierApi::Handler);
}

// tests-10-02 U15, step 4, queue barrier control.
TEST_F(ReusableEventsSchedulerTest, QueueBarrierBridgesForeignSignal) {
  checkBarrierBridgesForeignSignal(BarrierApi::Queue);
}

// tests-10-02 U15, step 5; extends HeldBarrierWaitsForCapturedSignal.
TEST_F(ReusableEventsSchedulerTest, HandlerBarrierFollowsTheCapturedSignal) {
  checkBarrierFollowsTheCapturedSignal(BarrierApi::Handler);
}

// tests-10-02 U15, step 5, queue barrier control.
TEST_F(ReusableEventsSchedulerTest, QueueBarrierFollowsTheCapturedSignal) {
  checkBarrierFollowsTheCapturedSignal(BarrierApi::Queue);
}

//===----------------------------------------------------------------------===//
// U16: host accessors.
//===----------------------------------------------------------------------===//

struct HostAccessorParams {
  const char *Name;
  bool GetHostAccess;
  bool Reassociate;
};

class ReusableEventsHostAccessorTest
    : public ReusableEventsSchedulerTest,
      public ::testing::WithParamInterface<HostAccessorParams> {};

INSTANTIATE_TEST_SUITE_P(
    ReusableEventsScheduler, ReusableEventsHostAccessorTest,
    ::testing::Values(
        HostAccessorParams{"HostAccessor", false, false},
        HostAccessorParams{"HostAccessorReassociated", false, true},
        HostAccessorParams{"GetHostAccess", true, false},
        HostAccessorParams{"GetHostAccessReassociated", true, true}),
    paramName<HostAccessorParams>);

// tests-10-02 U16. A signal behind a kernel blocked by a host accessor, and
// its consumers on another queue, stay in the scheduler while any copy of
// the accessor exists; releasing it lets the kernel in, but does not complete
// the signal. The mock does not run kernels, so the data ordering is not
// checked; host image accessors are not covered.
TEST_P(ReusableEventsHostAccessorTest, HoldsTheSignalAndItsConsumers) {
  const HostAccessorParams &P = GetParam();
  int Data[4] = {};
  {
    sycl::buffer<int, 1> Buf{Data, sycl::range<1>{4}};
    sycl::queue Q1 = inOrderQueue();
    sycl::queue Q2 = inOrderQueue();
    sycl::queue Q3 = inOrderQueue();
    sycl::event E = syclex::make_event(Ctx);
    std::future<void> Drain;
    CompleteAllAtScopeExit CompleteAll;
    std::optional<sycl::host_accessor<int, 1>> First;
    std::optional<sycl::host_accessor<int, 1>> Second;

    if (P.GetHostAccess)
      First.emplace(Buf.get_host_access());
    else
      First.emplace(Buf);
    Second.emplace(*First);

    const size_t Launches = kernelLaunches();
    const size_t Created = createdEvents();
    writeKernel(Q1, Buf);
    syclex::enqueue_signal_event(Q1, E);
    const std::shared_ptr<sycl::detail::event_binding> Original = bindingOf(E);
    syclex::enqueue_wait_event(Q2, E);
    submitKernel(Q2, {E});
    ur_event_handle_t Reassociated = nullptr;
    if (P.Reassociate) {
      syclex::enqueue_signal_event(Q3, E);
      Reassociated = handleOf(E);
      EXPECT_TRUE(isComplete(E));
    }

    // Neither the signal nor its consumers bypass the accessor.
    EXPECT_EQ(kernelLaunches(), Launches);
    EXPECT_EQ(Original->getHandle(), nullptr);
    if (!P.Reassociate) {
      EXPECT_EQ(createdEvents(), Created);
    }

    // Another copy of the accessor still holds the buffer.
    First.reset();
    EXPECT_EQ(kernelLaunches(), Launches);
    EXPECT_EQ(Original->getHandle(), nullptr);

    // Releasing the last copy lets the kernel in; the signal waits for the
    // kernel's backend event.
    setKernelsStayPending(true);
    setBarriersStayPending(true);
    Second.reset();
    EXPECT_TRUE(eventually(
        [&] { return KernelLaunchWaitLists.size() == Launches + 1; }));
    EXPECT_FALSE(Original->isCompleted());

    // The consumer is submitted behind the pending signal.
    setKernelsStayPending(false);
    Drain = std::async(std::launch::async, [&] { Q2.wait(); });
    EXPECT_TRUE(eventually(
        [&] { return KernelLaunchWaitLists.size() == Launches + 2; }));
    const ur_event_handle_t Signal = Original->getHandle();
    ASSERT_NE(Signal, nullptr);
    const std::vector<ur_event_handle_t> Consumer =
        kernelWaitList(Launches + 1);
    EXPECT_TRUE(contains(Consumer, Signal));
    if (Reassociated) {
      EXPECT_FALSE(contains(Consumer, Reassociated));
    }
    completeAll();
    EXPECT_TRUE(finishes(Drain));
    Q1.wait();
  }
}

//===----------------------------------------------------------------------===//
// U17: memory-manager commands.
//===----------------------------------------------------------------------===//

struct Buffers {
  sycl::buffer<int, 1> &Buf;
  sycl::buffer<int, 1> &Other;
  sycl::buffer<int, 1> &Sub;
  int *Host;
};

struct MemoryCommand {
  const char *Name;
  std::function<void(sycl::handler &, Buffers &)> Record;
};

class ReusableEventsMemoryCommandTest
    : public ReusableEventsSchedulerTest,
      public ::testing::WithParamInterface<MemoryCommand> {};

INSTANTIATE_TEST_SUITE_P(
    ReusableEventsScheduler, ReusableEventsMemoryCommandTest,
    ::testing::Values(
        MemoryCommand{"Fill",
                      [](sycl::handler &CGH, Buffers &B) {
                        sycl::accessor Acc{B.Buf, CGH, sycl::write_only};
                        CGH.fill(Acc, 1);
                      }},
        MemoryCommand{"SubBufferFill",
                      [](sycl::handler &CGH, Buffers &B) {
                        sycl::accessor Acc{B.Sub, CGH, sycl::write_only};
                        CGH.fill(Acc, 1);
                      }},
        MemoryCommand{"CopyToHost",
                      [](sycl::handler &CGH, Buffers &B) {
                        sycl::accessor Acc{B.Buf, CGH, sycl::read_only};
                        CGH.copy(Acc, B.Host);
                      }},
        MemoryCommand{"CopyFromHost",
                      [](sycl::handler &CGH, Buffers &B) {
                        sycl::accessor Acc{B.Buf, CGH, sycl::write_only};
                        CGH.copy(static_cast<const int *>(B.Host), Acc);
                      }},
        MemoryCommand{"CopyBetweenBuffers",
                      [](sycl::handler &CGH, Buffers &B) {
                        sycl::accessor Src{B.Other, CGH, sycl::read_only};
                        sycl::accessor Dst{B.Buf, CGH, sycl::write_only};
                        CGH.copy(Src, Dst);
                      }}),
    paramName<MemoryCommand>);
// update_host is not covered: addCGUpdateHost drops the command group's event
// dependencies, so it never captures the signal.

// tests-10-02 U17. A buffer command held behind a host task waits for the
// signal it captured, not the newer one which completed first, and the
// buffers are released as often as they were created. The completed
// no-handle allocation dependency of step 2 is not covered.
TEST_P(ReusableEventsMemoryCommandTest, WaitsForTheCapturedSignal) {
  ur_event_handle_t First = nullptr;
  ur_event_handle_t Second = nullptr;
  {
    int Data[4] = {};
    int OtherData[4] = {};
    int Host[4] = {};
    sycl::buffer<int, 1> Buf{Data, sycl::range<1>{4}};
    sycl::buffer<int, 1> Other{OtherData, sycl::range<1>{4}};
    sycl::buffer<int, 1> Sub{Buf, sycl::id<1>{0}, sycl::range<1>{2}};
    Buffers B{Buf, Other, Sub, Host};
    sycl::queue SignalQ = inOrderQueue();
    sycl::queue ResignalQ = inOrderQueue();
    sycl::queue Q = inOrderQueue();
    const ur_queue_handle_t QHandle = handleOf(Q);
    sycl::event E = syclex::make_event(Ctx);
    auto Gate = std::make_shared<HostTaskGate>();
    CompleteAllAtScopeExit CompleteAll;
    OpenAtScopeExit OpenGate{Gate};

    signalPending(SignalQ, E);
    First = handleOf(E);
    blockQueue(Q, Gate);
    Q.submit([&](sycl::handler &CGH) {
      CGH.depends_on(E);
      GetParam().Record(CGH, B);
    });

    // The newer signal completes first.
    signalPending(ResignalQ, E);
    Second = handleOf(E);
    ASSERT_NE(First, nullptr);
    ASSERT_NE(Second, nullptr);
    ASSERT_NE(First, Second);
    complete(Second);
    EXPECT_FALSE(inSomeWaitList(First));

    Gate->open();
    EXPECT_TRUE(eventually([&] {
      return std::any_of(
          MemoryCalls.begin(), MemoryCalls.end(), [&](const MemoryCall &Call) {
            return Call.Queue == QHandle && contains(Call.WaitList, First);
          });
    })) << "the buffer command did not wait for the captured signal";
    EXPECT_FALSE(inSomeWaitList(Second));
    complete(First);
    Q.wait();
  }
  EXPECT_TRUE(
      eventually([] { return MemCreates + MemRetains == MemReleases; }));
}

// tests-10-02 U17, step 4: a buffer command on a signal of another context
// waits for it on the host.
TEST_F(ReusableEventsSchedulerTest, MemoryCommandBridgesForeignSignal) {
  sycl::context OtherCtx{Dev};
  {
    int Data[4] = {};
    sycl::buffer<int, 1> Buf{Data, sycl::range<1>{4}};
    sycl::queue SignalQ = inOrderQueue(OtherCtx);
    sycl::queue Q = inOrderQueue();
    const ur_queue_handle_t QHandle = handleOf(Q);
    track(SignalQ);
    track(Q);
    sycl::event E = syclex::make_event(OtherCtx);
    CompleteAllAtScopeExit CompleteAll;

    signalPending(SignalQ, E);
    const ur_event_handle_t Signal = handleOf(E);
    ASSERT_NE(Signal, nullptr);
    Q.submit([&](sycl::handler &CGH) {
      CGH.depends_on(E);
      sycl::accessor Acc{Buf, CGH, sycl::write_only};
      CGH.fill(Acc, 1);
    });
    EXPECT_TRUE(eventually([&] { return contains(WaitedEvents, Signal); }));
    complete(Signal);
    Q.wait();
    EXPECT_TRUE(eventually([&] {
      return std::any_of(
          MemoryCalls.begin(), MemoryCalls.end(),
          [&](const MemoryCall &Call) { return Call.Queue == QHandle; });
    }));
    EXPECT_FALSE(inSomeWaitList(Signal));
    EXPECT_FALSE(foreignHandleSubmitted());
  }
  EXPECT_TRUE(
      eventually([] { return MemCreates + MemRetains == MemReleases; }));
}

//===----------------------------------------------------------------------===//
// U18, U19: cross-context waits.
//===----------------------------------------------------------------------===//

// tests-10-02 U18, steps 1-4. Two adapters (step 5) are not covered: the mock
// has one.
TEST_F(ReusableEventsSchedulerTest, ForeignConsumerFollowsTheOriginalSignal) {
  sycl::context Other{Dev};
  ur_event_handle_t First = nullptr;
  ur_event_handle_t Second = nullptr;
  {
    sycl::queue SignalQ = inOrderQueue();
    sycl::queue ResignalQ = inOrderQueue();
    sycl::queue Q = inOrderQueue(Other);
    track(SignalQ);
    track(ResignalQ);
    track(Q);
    sycl::event E = syclex::make_event(Ctx);
    std::future<void> Drain;
    CompleteAllAtScopeExit CompleteAll;

    const size_t Launches = kernelLaunches();
    signalPending(SignalQ, E);
    First = handleOf(E);
    ASSERT_NE(First, nullptr);
    syclex::enqueue_wait_event(Q, E);
    submitKernel(Q);
    // The other context waits for the signal on the host.
    EXPECT_TRUE(eventually([&] { return contains(WaitedEvents, First); }));

    signalPending(ResignalQ, E);
    Second = handleOf(E);
    ASSERT_NE(Second, nullptr);
    ASSERT_NE(First, Second);
    complete(Second);
    Drain = std::async(std::launch::async, [&] { Q.wait(); });
    EXPECT_TRUE(stillBlocked(Drain));
    EXPECT_EQ(kernelLaunches(), Launches);

    complete(First);
    EXPECT_TRUE(finishes(Drain));
    EXPECT_EQ(kernelLaunches(), Launches + 1);
    EXPECT_FALSE(foreignHandleSubmitted());
  }
  EXPECT_TRUE(eventually([&] {
    return ownershipBalancedLocked(First) && ownershipBalancedLocked(Second);
  }));
}

struct ChainParams {
  const char *Name;
  bool Native;
  bool HostTaskMiddle;
};

class ReusableEventsCrossContextChainTest
    : public ReusableEventsSchedulerTest,
      public ::testing::WithParamInterface<ChainParams> {
protected:
  bool nativeSupport() const override { return GetParam().Native; }
};

INSTANTIATE_TEST_SUITE_P(
    ReusableEventsScheduler, ReusableEventsCrossContextChainTest,
    ::testing::Values(ChainParams{"KernelNativeSupport", true, false},
                      ChainParams{"HostTaskNativeSupport", true, true},
                      ChainParams{"KernelNoNativeSupport", false, false},
                      ChainParams{"HostTaskNoNativeSupport", false, true}),
    paramName<ChainParams>);

// tests-10-02 U19. A chain C1 signal -> C2 wait and work -> C2 signal -> C1
// wait and kernel is submitted without waiting for the held producer, and
// advances once the producer is released. Whether a scheduler lock is held
// across a host wait (step 4) is not observable here.
TEST_P(ReusableEventsCrossContextChainTest, AdvancesWithoutDeadlock) {
  const ChainParams &P = GetParam();
  sycl::context Other{Dev};
  {
    sycl::queue Producer = inOrderQueue();
    sycl::queue Resignal = inOrderQueue();
    sycl::queue Middle = inOrderQueue(Other);
    sycl::queue Last = inOrderQueue();
    track(Producer);
    track(Resignal);
    track(Middle);
    track(Last);
    sycl::event E1 = syclex::make_event(Ctx);
    sycl::event E2 = syclex::make_event(Other);
    auto Gate = std::make_shared<HostTaskGate>();
    std::future<void> Submit;
    std::future<void> Drain;
    CompleteAllAtScopeExit CompleteAll;
    OpenAtScopeExit OpenGate{Gate};

    const size_t Launches = kernelLaunches();
    blockQueue(Producer, Gate);
    Submit = std::async(std::launch::async, [&] {
      syclex::enqueue_signal_event(Producer, E1);
      syclex::enqueue_wait_event(Middle, E1);
      if (P.HostTaskMiddle)
        Middle.submit([](sycl::handler &CGH) { CGH.host_task([] {}); });
      else
        submitKernel(Middle);
      syclex::enqueue_signal_event(Middle, E2);
      syclex::enqueue_wait_event(Last, E2);
      submitKernel(Last);
    });
    // Submission does not wait for the gate.
    ASSERT_TRUE(finishes(Submit));

    // The middle wait captured E1; signal it again.
    syclex::enqueue_signal_event(Resignal, E1);
    EXPECT_EQ(kernelLaunches(), Launches);

    Gate->open();
    Drain = std::async(std::launch::async, [&] { Last.wait(); });
    EXPECT_TRUE(finishes(Drain));
    EXPECT_EQ(kernelLaunches(), Launches + (P.HostTaskMiddle ? 1u : 2u));
    EXPECT_FALSE(foreignHandleSubmitted());
  }
}

//===----------------------------------------------------------------------===//
// U20: stream flushes and 2D host fallbacks.
//===----------------------------------------------------------------------===//

// Submits a kernel with a stream to the out-of-order queue Q. The kernel and
// the buffer commands of its stream flush stay pending until the test
// completes them.
sycl::event submitStreamKernel(sycl::queue &Q) {
  setKernelsStayPending(true);
  setMemoryOpsStayPending(true);
  sycl::event E = Q.submit([&](sycl::handler &CGH) {
    sycl::stream Out{1024, 256, CGH};
    CGH.single_task<BindingTestKernel>([] {});
  });
  setMemoryOpsStayPending(false);
  setKernelsStayPending(false);
  return E;
}

// tests-10-02 U20, steps 1-3: waiting for the original queue covers the
// kernel and its stream flush after the kernel's event moved on.
TEST_F(ReusableEventsSchedulerTest, QueueWaitCoversTheOriginalStreamFlush) {
  sycl::queue Q{Ctx, Dev};
  sycl::queue SignalQ = inOrderQueue();
  std::future<void> QueueWait;
  CompleteAllAtScopeExit CompleteAll;

  sycl::event EK = submitStreamKernel(Q);
  const ur_event_handle_t Kernel = handleOf(EK);
  ASSERT_NE(Kernel, nullptr);
  const std::shared_ptr<sycl::detail::event_binding> Original = bindingOf(EK);
  syclex::enqueue_signal_event(SignalQ, EK);
  EXPECT_TRUE(isComplete(EK));

  QueueWait = std::async(std::launch::async, [&] { Q.wait(); });
  EXPECT_TRUE(stillBlocked(QueueWait));
  complete(Kernel);
  bool FlushHeld = false;
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    FlushHeld = !HeldMemoryEvents.empty();
  }
  if (FlushHeld) {
    EXPECT_TRUE(stillBlocked(QueueWait)) << "the flush was not waited for";
  }
  completeHeldMemoryOps();
  EXPECT_TRUE(finishes(QueueWait));
  EXPECT_TRUE(Original->isCompleted());
}

// tests-10-02 U20, step 2: waiting for the new signal does not wait for the
// stream flush of the old one.
// Known defect: review-10-01 #4 (the flush is attached to the event, not to
// the signal of the kernel, so every later signal waits for it).
TEST_F(ReusableEventsSchedulerTest,
       DISABLED_NewSignalDoesNotWaitForTheOldStreamFlush) {
  sycl::queue Q{Ctx, Dev};
  sycl::queue SignalQ = inOrderQueue();
  std::future<void> EventWait;
  CompleteAllAtScopeExit CompleteAll;

  sycl::event EK = submitStreamKernel(Q);
  const ur_event_handle_t Kernel = handleOf(EK);
  ASSERT_NE(Kernel, nullptr);
  syclex::enqueue_signal_event(SignalQ, EK);

  EventWait = std::async(std::launch::async, [&] { EK.wait(); });
  EXPECT_TRUE(finishes(EventWait));
  EXPECT_TRUE(isPending(Kernel));
  complete(Kernel);
  completeHeldMemoryOps();
  Q.wait();
}

struct Fallback2D {
  const char *Name;
  std::function<sycl::event(sycl::queue &, int *Src, int *Dst)> Submit;
  std::function<bool(const int *Src, const int *Dst)> Check;
};

constexpr size_t Width2D = 4;
constexpr size_t Height2D = 4;
constexpr size_t Pitch2D = 4;
constexpr size_t Size2D = Pitch2D * Height2D;

bool copied(const int *Src, const int *Dst) {
  return std::equal(Src, Src + Size2D, Dst);
}

bool filledWith(const int *Dst, int Value) {
  return std::all_of(Dst, Dst + Size2D, [&](int V) { return V == Value; });
}

int memsetValue() {
  int Value = 0;
  std::memset(&Value, 0x5a, sizeof(Value));
  return Value;
}

class ReusableEventsHostFallback2DTest
    : public ReusableEventsSchedulerTest,
      public ::testing::WithParamInterface<Fallback2D> {
protected:
  void SetUp() override {
    ReusableEventsSchedulerTest::SetUp();
    mock::getCallbacks().set_after_callback("urUSMGetMemAllocInfo",
                                            &afterUSMGetMemAllocInfo);
  }
};

INSTANTIATE_TEST_SUITE_P(
    ReusableEventsScheduler, ReusableEventsHostFallback2DTest,
    ::testing::Values(
        Fallback2D{"QueueMemcpy2D",
                   [](sycl::queue &Q, int *Src, int *Dst) {
                     return Q.ext_oneapi_memcpy2d(
                         Dst, Pitch2D * sizeof(int), Src, Pitch2D * sizeof(int),
                         Width2D * sizeof(int), Height2D);
                   },
                   copied},
        Fallback2D{"HandlerMemcpy2D",
                   [](sycl::queue &Q, int *Src, int *Dst) {
                     return Q.submit([=](sycl::handler &CGH) {
                       CGH.ext_oneapi_memcpy2d(Dst, Pitch2D * sizeof(int), Src,
                                               Pitch2D * sizeof(int),
                                               Width2D * sizeof(int), Height2D);
                     });
                   },
                   copied},
        Fallback2D{"QueueCopy2D",
                   [](sycl::queue &Q, int *Src, int *Dst) {
                     return Q.ext_oneapi_copy2d(static_cast<const int *>(Src),
                                                Pitch2D, Dst, Pitch2D, Width2D,
                                                Height2D);
                   },
                   copied},
        Fallback2D{"HandlerCopy2D",
                   [](sycl::queue &Q, int *Src, int *Dst) {
                     return Q.submit([=](sycl::handler &CGH) {
                       CGH.ext_oneapi_copy2d(static_cast<const int *>(Src),
                                             Pitch2D, Dst, Pitch2D, Width2D,
                                             Height2D);
                     });
                   },
                   copied},
        Fallback2D{
            "QueueFill2D",
            [](sycl::queue &Q, int *, int *Dst) {
              return Q.ext_oneapi_fill2d(Dst, Pitch2D, 9, Width2D, Height2D);
            },
            [](const int *, const int *Dst) { return filledWith(Dst, 9); }},
        Fallback2D{
            "HandlerFill2D",
            [](sycl::queue &Q, int *, int *Dst) {
              return Q.submit([=](sycl::handler &CGH) {
                CGH.ext_oneapi_fill2d(Dst, Pitch2D, 9, Width2D, Height2D);
              });
            },
            [](const int *, const int *Dst) { return filledWith(Dst, 9); }},
        Fallback2D{"QueueMemset2D",
                   [](sycl::queue &Q, int *, int *Dst) {
                     return Q.ext_oneapi_memset2d(Dst, Pitch2D * sizeof(int),
                                                  0x5a, Width2D * sizeof(int),
                                                  Height2D);
                   },
                   [](const int *, const int *Dst) {
                     return filledWith(Dst, memsetValue());
                   }},
        Fallback2D{"HandlerMemset2D",
                   [](sycl::queue &Q, int *, int *Dst) {
                     return Q.submit([=](sycl::handler &CGH) {
                       CGH.ext_oneapi_memset2d(Dst, Pitch2D * sizeof(int), 0x5a,
                                               Width2D * sizeof(int), Height2D);
                     });
                   },
                   [](const int *, const int *Dst) {
                     return filledWith(Dst, memsetValue());
                   }}),
    paramName<Fallback2D>);

// tests-10-02 U20, steps 4-5. A 2D operation on host memory runs as a host
// task; a later command captures its host completion. Its event cannot be
// signaled at 8337da70 (a host event; pinned). The device-target fill with
// backend support off falls back to a kernel without kernel info in the mock,
// and is not covered.
TEST_P(ReusableEventsHostFallback2DTest, LaterCommandsWaitForTheHostWork) {
  std::vector<int> Src(Size2D);
  std::vector<int> Dst(Size2D, 0);
  std::iota(Src.begin(), Src.end(), 1);
  {
    sycl::queue Q = inOrderQueue();
    sycl::queue ConsumerQ = inOrderQueue();
    sycl::queue SignalQ = inOrderQueue();
    auto Gate = std::make_shared<HostTaskGate>();
    CompleteAllAtScopeExit CompleteAll;
    OpenAtScopeExit OpenGate{Gate};

    const size_t Launches = kernelLaunches();
    blockQueue(Q, Gate);
    sycl::event Fallback = GetParam().Submit(Q, Src.data(), Dst.data());
    submitKernel(ConsumerQ, {Fallback});
    try {
      syclex::enqueue_signal_event(SignalQ, Fallback);
      ADD_FAILURE() << "a host task event was signaled";
    } catch (const sycl::exception &Ex) {
      EXPECT_EQ(Ex.code(), sycl::make_error_code(sycl::errc::invalid));
    }
    EXPECT_EQ(kernelLaunches(), Launches);

    Gate->open();
    Q.wait();
    ConsumerQ.wait();
    EXPECT_EQ(kernelLaunches(), Launches + 1);
    EXPECT_TRUE(GetParam().Check(Src.data(), Dst.data()));
  }
}

//===----------------------------------------------------------------------===//
// U21: auxiliary resources.
//===----------------------------------------------------------------------===//

// tests-10-02 U21. Resources registered for a kernel - a counted object and a
// device allocation, as for a reduction - live until the kernel completes,
// even after its event moved on to a completed signal. With Transfer, they are
// moved to the kernel from another command, as along a reduction's chain.
void checkResourcesLiveUntilTheOriginalCompletes(const sycl::context &Ctx,
                                                 const sycl::device &Dev,
                                                 bool Transfer) {
  int *Memory = sycl::malloc_device<int>(4, Dev, Ctx);
  ASSERT_NE(Memory, nullptr);
  auto FreesOfMemory = [&] {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    return std::count(FreedPointers.begin(), FreedPointers.end(),
                      static_cast<void *>(Memory));
  };
  {
    sycl::queue Q{Ctx, Dev};
    sycl::queue SignalQ{Ctx, Dev, sycl::property::queue::in_order{}};
    std::future<void> Cleanup;
    CompleteAllAtScopeExit CompleteAll;

    setKernelsStayPending(true);
    sycl::event E = submitKernel(Q);
    sycl::event Source = submitKernel(Q);
    setKernelsStayPending(false);
    const ur_event_handle_t Kernel = handleOf(E);
    ASSERT_NE(Kernel, nullptr);
    // Keep the kernel's signal referenced, so that the next signal of E gets
    // a binding of its own.
    const std::shared_ptr<sycl::detail::event_binding> Original = bindingOf(E);

    {
      sycl::context C = Ctx;
      std::vector<std::shared_ptr<const void>> Resources{
          std::shared_ptr<int>(new int(0),
                               [](int *P) {
                                 ++DeletedResources;
                                 delete P;
                               }),
          std::shared_ptr<int>(Memory, [C](int *P) { sycl::free(P, C); })};
      if (Transfer) {
        SchedulerAccess::registerResources(sycl::detail::getSyclObjImpl(Source),
                                           std::move(Resources));
        SchedulerAccess::takeResources(sycl::detail::getSyclObjImpl(E),
                                       sycl::detail::getSyclObjImpl(Source));
      } else {
        SchedulerAccess::registerResources(sycl::detail::getSyclObjImpl(E),
                                           std::move(Resources));
      }
    }
    complete(handleOf(Source));

    syclex::enqueue_signal_event(SignalQ, E);
    ASSERT_TRUE(isComplete(E));
    SchedulerAccess::cleanupResources(sycl::detail::BlockingT::NON_BLOCKING);
    EXPECT_EQ(DeletedResources, 0);
    EXPECT_EQ(FreesOfMemory(), 0);

    Cleanup = std::async(std::launch::async, [] {
      SchedulerAccess::cleanupResources(sycl::detail::BlockingT::BLOCKING);
    });
    EXPECT_TRUE(stillBlocked(Cleanup))
        << "the blocking cleanup did not wait for the kernel";
    complete(Kernel);
    EXPECT_TRUE(finishes(Cleanup));
    EXPECT_EQ(DeletedResources, 1);
    EXPECT_EQ(FreesOfMemory(), 1);
  }
  // Leave nothing registered with the scheduler.
  SchedulerAccess::cleanupResources(sycl::detail::BlockingT::BLOCKING);
}

// tests-10-02 U21, steps 1-5.
// Known defect: review-10-02 #3 (auxiliary resources are keyed by the event,
// so the cleanup follows its latest signal).
TEST_F(ReusableEventsSchedulerTest,
       DISABLED_AuxiliaryResourcesLiveUntilTheOriginalCompletes) {
  checkResourcesLiveUntilTheOriginalCompletes(Ctx, Dev, /*Transfer*/ false);
}

// tests-10-02 U21, step 5, takeAuxiliaryResources.
// Known defect: review-10-02 #3 (auxiliary resources are keyed by the event,
// so the cleanup follows its latest signal).
TEST_F(ReusableEventsSchedulerTest,
       DISABLED_TransferredAuxiliaryResourcesLiveUntilTheOriginalCompletes) {
  checkResourcesLiveUntilTheOriginalCompletes(Ctx, Dev, /*Transfer*/ true);
}

//===----------------------------------------------------------------------===//
// U22: buffer record cleanup.
//===----------------------------------------------------------------------===//

// tests-10-02 U22, steps 1-3: a kernel writing a buffer remains an incomplete
// leaf of the buffer's record after its event moved on to a completed signal,
// and the buffer's destruction waits for it.
// Known defect: review-10-01 #3 (checkLeavesCompletion asks the command's
// event, which reports its latest signal).
TEST_F(ReusableEventsSchedulerTest,
       DISABLED_ReSignaledKernelRemainsAnIncompleteLeaf) {
  int Data[4] = {1, 2, 3, 4};
  sycl::queue Q{Ctx, Dev};
  sycl::queue SignalQ = inOrderQueue();
  std::optional<sycl::buffer<int, 1>> Buf;
  Buf.emplace(Data, sycl::range<1>{4});
  std::future<void> Destroy;
  CompleteAllAtScopeExit CompleteAll;

  setKernelsStayPending(true);
  sycl::event E = writeKernel(Q, *Buf);
  setKernelsStayPending(false);
  const ur_event_handle_t Kernel = handleOf(E);
  ASSERT_NE(Kernel, nullptr);
  const std::shared_ptr<sycl::detail::event_binding> Original = bindingOf(E);
  syclex::enqueue_signal_event(SignalQ, E);
  ASSERT_TRUE(isComplete(E));

  sycl::detail::MemObjRecord *Record = recordOf(*Buf);
  ASSERT_NE(Record, nullptr);
  EXPECT_FALSE(SchedulerAccess::leavesCompleted(Record));

  Destroy = std::async(std::launch::async, [&] { Buf.reset(); });
  EXPECT_TRUE(stillBlocked(Destroy))
      << "the buffer was destroyed before the kernel completed";
  complete(Kernel);
  EXPECT_TRUE(finishes(Destroy));
}

// tests-10-02 U22, step 2, controls: a pending leaf is incomplete, a completed
// one is complete.
TEST_F(ReusableEventsSchedulerTest, LeavesCompletionFollowsTheKernel) {
  int Data[4] = {1, 2, 3, 4};
  sycl::queue Q{Ctx, Dev};
  sycl::buffer<int, 1> Buf{Data, sycl::range<1>{4}};
  CompleteAllAtScopeExit CompleteAll;

  setKernelsStayPending(true);
  sycl::event E = writeKernel(Q, Buf);
  setKernelsStayPending(false);
  const ur_event_handle_t Kernel = handleOf(E);
  ASSERT_NE(Kernel, nullptr);

  sycl::detail::MemObjRecord *Record = recordOf(Buf);
  ASSERT_NE(Record, nullptr);
  EXPECT_FALSE(SchedulerAccess::leavesCompleted(Record));
  complete(Kernel);
  E.wait();
  EXPECT_TRUE(SchedulerAccess::leavesCompleted(Record));
}

// tests-10-02 U22, step 4: the copy back of a buffer waits for the original
// kernel after its event moved on, and precedes the release of the buffer.
// The mock does not copy data, so the host data is not checked; neither is
// a host or no-handle leaf (step 5).
TEST_F(ReusableEventsSchedulerTest, CopyBackFollowsTheOriginalKernel) {
  int Data[4] = {1, 2, 3, 4};
  size_t LogBase = 0;
  size_t CallBase = 0;
  ur_event_handle_t Kernel = nullptr;
  {
    sycl::queue Q{Ctx, Dev};
    sycl::queue SignalQ = inOrderQueue();
    std::optional<sycl::buffer<int, 1>> Buf;
    Buf.emplace(Data, sycl::range<1>{4});
    std::future<void> Destroy;
    CompleteAllAtScopeExit CompleteAll;

    setKernelsStayPending(true);
    sycl::event E = writeKernel(Q, *Buf);
    setKernelsStayPending(false);
    Kernel = handleOf(E);
    ASSERT_NE(Kernel, nullptr);
    const std::shared_ptr<sycl::detail::event_binding> Original = bindingOf(E);
    syclex::enqueue_signal_event(SignalQ, E);

    {
      std::lock_guard<std::mutex> Lock(BackendMutex);
      LogBase = MemoryLog.size();
      CallBase = MemoryCalls.size();
    }
    Destroy = std::async(std::launch::async, [&] { Buf.reset(); });
    EXPECT_TRUE(eventually([&] {
      return std::any_of(MemoryCalls.begin() + CallBase, MemoryCalls.end(),
                         [&](const MemoryCall &Call) {
                           return contains(Call.WaitList, Kernel);
                         });
    })) << "the copy back did not wait for the original kernel";
    complete(Kernel);
    EXPECT_TRUE(finishes(Destroy));
  }
  std::lock_guard<std::mutex> Lock(BackendMutex);
  auto Begin = MemoryLog.begin() + LogBase;
  auto FirstRelease = std::find(Begin, MemoryLog.end(), "Release");
  auto FirstCommand =
      std::find_if(Begin, MemoryLog.end(),
                   [](const std::string &N) { return N != "Release"; });
  EXPECT_NE(FirstCommand, MemoryLog.end());
  EXPECT_TRUE(FirstCommand < FirstRelease)
      << "the buffer was released before it was copied back";
}

//===----------------------------------------------------------------------===//
// U27: profiling.
//===----------------------------------------------------------------------===//

struct ProfilingParams {
  const char *Name;
  bool PerEvent;
};

class ReusableEventsProfilingTest
    : public ReusableEventsSchedulerTest,
      public ::testing::WithParamInterface<ProfilingParams> {
protected:
  void SetUp() override {
    ReusableEventsSchedulerTest::SetUp();
    mock::getCallbacks().set_replace_callback("urEventGetProfilingInfo",
                                              &replaceEventGetProfilingInfo);
  }

  // A queue to signal on: with profiling, unless the event has it.
  sycl::queue signalQueue() {
    if (GetParam().PerEvent)
      return inOrderQueue();
    return sycl::queue{
        Ctx, Dev,
        sycl::property_list{sycl::property::queue::in_order{},
                            sycl::property::queue::enable_profiling{}}};
  }

  sycl::event makeEvent() {
    if (GetParam().PerEvent)
      return syclex::make_event(
          Ctx, syclex::properties{syclex::enable_profiling{true}});
    return syclex::make_event(Ctx);
  }
};

INSTANTIATE_TEST_SUITE_P(
    ReusableEventsScheduler, ReusableEventsProfilingTest,
    ::testing::Values(ProfilingParams{"PerEventProfiling", true},
                      ProfilingParams{"ProfilingQueue", false}),
    paramName<ProfilingParams>);

std::pair<uint64_t, uint64_t> startAndEnd(const sycl::event &E) {
  return {E.get_profiling_info<sycl::info::event_profiling::command_start>(),
          E.get_profiling_info<sycl::info::event_profiling::command_end>()};
}

// tests-10-02 U27, step 4, control: a completed signal reports its own
// timestamps; a signal is a profiling tag, with equal start and end.
TEST_P(ReusableEventsProfilingTest, CompletedSignalsReportTheirOwnProfiling) {
  sycl::queue Q = signalQueue();
  sycl::queue Other = signalQueue();
  sycl::queue ConsumerQ = inOrderQueue();
  sycl::event E = makeEvent();
  auto Gate = std::make_shared<HostTaskGate>();
  CompleteAllAtScopeExit CompleteAll;
  OpenAtScopeExit OpenGate{Gate};

  syclex::enqueue_signal_event(Q, E);
  const ur_event_handle_t First = handleOf(E);
  ASSERT_NE(First, nullptr);
  const uint64_t FirstEnd =
      profilingValue(First, UR_PROFILING_INFO_COMMAND_END);
  EXPECT_EQ(startAndEnd(E), std::make_pair(FirstEnd, FirstEnd));

  blockQueue(ConsumerQ, Gate);
  submitKernel(ConsumerQ, {E});
  syclex::enqueue_signal_event(Other, E);
  const ur_event_handle_t Second = handleOf(E);
  ASSERT_NE(Second, nullptr);
  ASSERT_NE(Second, First);
  const uint64_t SecondEnd =
      profilingValue(Second, UR_PROFILING_INFO_COMMAND_END);
  EXPECT_EQ(startAndEnd(E), std::make_pair(SecondEnd, SecondEnd));
}

// tests-10-02 U27, steps 1-4. A query for a signal held in the scheduler
// waits for it, and reports neither the previous signal's timestamps nor
// zeros. Host-task profiling and native support off (step 5) are not covered.
// Investigation (tests-10-02 U27): pins behaviour at 8337da70; oracle not
// agreed.
TEST_P(ReusableEventsProfilingTest, DeferredQueryWaitsForTheNewSignal) {
  sycl::queue OldQ = signalQueue();
  sycl::queue SignalQ = signalQueue();
  sycl::queue ConsumerQ = inOrderQueue();
  sycl::event E = makeEvent();
  auto ConsumerGate = std::make_shared<HostTaskGate>();
  auto SignalGate = std::make_shared<HostTaskGate>();
  std::future<std::pair<uint64_t, uint64_t>> Query;
  CompleteAllAtScopeExit CompleteAll;
  OpenAtScopeExit OpenConsumer{ConsumerGate};
  OpenAtScopeExit OpenSignal{SignalGate};

  signalPending(OldQ, E);
  const ur_event_handle_t First = handleOf(E);
  ASSERT_NE(First, nullptr);
  blockQueue(ConsumerQ, ConsumerGate);
  submitKernel(ConsumerQ, {E});

  blockQueue(SignalQ, SignalGate);
  syclex::enqueue_signal_event(SignalQ, E);
  complete(First);

  Query = std::async(std::launch::async, [E] { return startAndEnd(E); });
  EXPECT_TRUE(stillBlocked(Query));

  SignalGate->open();
  ASSERT_TRUE(finishes(Query));
  const ur_event_handle_t Second = handleOf(E);
  ASSERT_NE(Second, nullptr);
  ASSERT_NE(Second, First);
  const uint64_t SecondEnd =
      profilingValue(Second, UR_PROFILING_INFO_COMMAND_END);
  const std::pair<uint64_t, uint64_t> Result = Query.get();
  EXPECT_EQ(Result, std::make_pair(SecondEnd, SecondEnd));
  EXPECT_NE(Result.second,
            profilingValue(First, UR_PROFILING_INFO_COMMAND_END));
}

//===----------------------------------------------------------------------===//
// U29: failed signals.
//===----------------------------------------------------------------------===//

// tests-10-02 U29, native event creation fails.
TEST_F(ReusableEventsSchedulerTest, FailedEventCreationWakesWaiters) {
  checkFailedSignal(FailurePoint::EventCreate);
}

// tests-10-02 U29, the prerequisite events wait fails. Best effort: see
// checkFailedSignal.
TEST_F(ReusableEventsSchedulerTest, FailedEventsWaitWakesWaiters) {
  checkFailedSignal(FailurePoint::EventsWait);
}

// tests-10-02 U29, the barrier fails after the native event was created.
// Known defect: review-10-01 #4 (the created handle stays published without
// completion: waiters block on it and it leaks).
TEST_F(ReusableEventsSchedulerTest,
       DISABLED_FailedBarrierWakesWaitersAndReleasesTheCreatedEvent) {
  checkFailedSignal(FailurePoint::Barrier);
}

// tests-10-02 U29, the barrier fails without native support, i.e. with a
// backend-created output event.
TEST_F(ReusableEventsSchedulerNoNativeTest, FailedBarrierWakesWaiters) {
  checkFailedSignal(FailurePoint::Barrier);
}

//===----------------------------------------------------------------------===//
// U30: host task failures and concurrent completion.
//===----------------------------------------------------------------------===//

// tests-10-02 U30, steps 1-4: a failing host task releases every dependent
// root exactly once, and its error reaches its own queue only.
TEST_F(ReusableEventsSchedulerTest, HostTaskFailureWakesEveryDependent) {
  AsyncErrors Errors1, Errors2, Errors3, Errors4;
  {
    sycl::queue Q1{Ctx, Dev, Errors1.handler(),
                   sycl::property::queue::in_order{}};
    sycl::queue Q2{Ctx, Dev, Errors2.handler(),
                   sycl::property::queue::in_order{}};
    sycl::queue Q3{Ctx, Dev, Errors3.handler(),
                   sycl::property::queue::in_order{}};
    sycl::queue Q4{Ctx, Dev, Errors4.handler(),
                   sycl::property::queue::in_order{}};
    sycl::event E = syclex::make_event(Ctx);
    auto Gate = std::make_shared<HostTaskGate>();
    std::future<bool> SignalWaiter;
    std::future<bool> KernelWaiter;
    std::future<void> WaitWaiter;
    CompleteAllAtScopeExit CompleteAll;
    OpenAtScopeExit OpenGate{Gate};

    const size_t Launches = kernelLaunches();
    sycl::event HostTask = Q1.submit([&](sycl::handler &CGH) {
      CGH.host_task([Gate] {
        Gate->wait();
        throw std::runtime_error("host task failure");
      });
    });
    syclex::enqueue_signal_event(Q1, E);
    Q2.ext_oneapi_submit_barrier(std::vector<sycl::event>{HostTask});
    syclex::enqueue_signal_event(Q2, E);
    sycl::event Kernel = submitKernel(Q3, {HostTask});
    syclex::enqueue_wait_event(Q4, E);
    EXPECT_EQ(createdEvents(), 0u);

    SignalWaiter =
        std::async(std::launch::async, [E] { return waitQuietly(E); });
    KernelWaiter = std::async(std::launch::async,
                              [Kernel] { return waitQuietly(Kernel); });
    WaitWaiter = std::async(std::launch::async, [&] { Q4.wait(); });
    Gate->open();
    EXPECT_TRUE(finishes(SignalWaiter));
    EXPECT_TRUE(finishes(KernelWaiter));
    EXPECT_TRUE(finishes(WaitWaiter));
    Q1.wait();
    Q2.wait();
    Q3.wait();

    EXPECT_EQ(kernelLaunches(), Launches + 1);
    {
      // Each signal reached the backend once.
      std::lock_guard<std::mutex> Lock(BackendMutex);
      EXPECT_EQ(CreatedEvents.size(), 2u);
      for (ur_event_handle_t Handle : CreatedEvents)
        EXPECT_EQ(std::count(BarrierOutEvents.begin(), BarrierOutEvents.end(),
                             Handle),
                  1);
    }
    Q1.throw_asynchronous();
    Q2.throw_asynchronous();
    Q3.throw_asynchronous();
    Q4.throw_asynchronous();
  }
  EXPECT_EQ(Errors1.count(), 1);
  EXPECT_EQ(Errors2.count(), 0);
  EXPECT_EQ(Errors3.count(), 0);
  EXPECT_EQ(Errors4.count(), 0);
}

// tests-10-02 U30, step 2, control: a dependent command registered while the
// host task completes is enqueued exactly once. Run repeatedly.
TEST_F(ReusableEventsSchedulerTest, HostTaskCompletionRacesADependent) {
  sycl::queue Q1 = inOrderQueue();
  sycl::queue Q3 = inOrderQueue();
  sycl::event E = syclex::make_event(Ctx);
  auto Gate = std::make_shared<HostTaskGate>();
  std::future<sycl::event> Submitter;
  CompleteAllAtScopeExit CompleteAll;
  OpenAtScopeExit OpenGate{Gate};

  const size_t Launches = kernelLaunches();
  sycl::event HostTask = blockQueue(Q1, Gate);
  syclex::enqueue_signal_event(Q1, E);
  Submitter = std::async(std::launch::async,
                         [&] { return submitKernel(Q3, {HostTask}); });
  Gate->open();
  ASSERT_TRUE(finishes(Submitter));
  sycl::event Kernel = Submitter.get();
  Kernel.wait();
  E.wait();
  Q1.wait();
  Q3.wait();
  EXPECT_EQ(kernelLaunches(), Launches + 1);
  EXPECT_EQ(createdEvents(), 1u);
}

// tests-10-02 U30, step 5, direct event: a failed backend wait is an error.
TEST_F(ReusableEventsSchedulerTest, FailedEventWaitThrows) {
  sycl::queue Q = inOrderQueue();
  sycl::event E = syclex::make_event(Ctx);
  CompleteAllAtScopeExit CompleteAll;

  signalPending(Q, E);
  const ur_event_handle_t Signal = handleOf(E);
  ASSERT_NE(Signal, nullptr);
  setFailingWait(Signal);
  EXPECT_THROW(E.wait(), sycl::exception);
  EXPECT_EQ(injectedFailures(), 1);
  setFailingWait(nullptr);
  complete(Signal);
  E.wait();
}

// tests-10-02 U30, step 5, captured dependency: a host task whose dependency
// wait fails does not run, and the error reaches its queue. A queue wait on a
// failing dependency is not covered.
TEST_F(ReusableEventsSchedulerTest, FailedDependencyWaitSkipsTheHostTask) {
  AsyncErrors HostTaskErrors;
  auto Ran = std::make_shared<std::atomic<bool>>(false);
  {
    sycl::queue KernelQ{Ctx, Dev};
    sycl::queue HostTaskQ{Ctx, Dev, HostTaskErrors.handler(),
                          sycl::property::queue::in_order{}};
    CompleteAllAtScopeExit CompleteAll;

    setKernelsStayPending(true);
    sycl::event Kernel = submitKernel(KernelQ);
    setKernelsStayPending(false);
    const ur_event_handle_t KernelHandle = handleOf(Kernel);
    ASSERT_NE(KernelHandle, nullptr);
    setFailingWait(KernelHandle);
    HostTaskQ.submit([&](sycl::handler &CGH) {
      CGH.depends_on(Kernel);
      CGH.host_task([Ran] { *Ran = true; });
    });
    EXPECT_TRUE(eventually([] { return InjectedFailures > 0; }));
    HostTaskQ.wait();
    EXPECT_FALSE(*Ran);
    HostTaskQ.throw_asynchronous();
    setFailingWait(nullptr);
    complete(KernelHandle);
    KernelQ.wait();
  }
  // The failed wait reports the backend error, and the host task adds its own
  // "Couldn't wait for host-task's dependencies".
  EXPECT_EQ(HostTaskErrors.count(), 2);
}

} // namespace
