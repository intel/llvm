//==-- ReusableEventsQueue.cpp --- Queue operations on reusable events -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The queue operations which take an event or hand one out: the public waits,
// the last-event query and the helpers built on it, external events, the
// flushes of the queue a consumed signal was submitted to, and the device a
// signal is attributed to. Each of them works on the signal the event
// represented when it was consumed, not on a later signal of the event.
//
// The scenario numbers refer to the reusable events test plan
// (tests-10-02.md); the findings to review-10-01.md and review-10-02.md.
//
//===----------------------------------------------------------------------===//

#include "ReusableEventsFixture.hpp"

#include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>

#include <optional>
#include <tuple>

namespace {

using namespace reusable_events_test;

using ReusableEventsQueueTest = ReusableEventsTest;

// Signals E on Q; the backend event of the signal stays pending until the test
// completes it.
void signalPending(sycl::queue &Q, sycl::event &E) {
  setBarriersStayPending(true);
  syclex::enqueue_signal_event(Q, E);
  setBarriersStayPending(false);
}

sycl::event submitKernel(sycl::queue &Q) {
  return Q.submit(
      [&](sycl::handler &CGH) { CGH.single_task<BindingTestKernel>([]() {}); });
}

sycl::event submitKernelAfter(sycl::queue &Q, const sycl::event &Dep) {
  return Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Dep);
    CGH.single_task<BindingTestKernel>([]() {});
  });
}

// Called with BackendMutex held, e.g. from an eventually predicate.
bool waitedForLocked(ur_event_handle_t Handle) {
  return std::find(WaitedEvents.begin(), WaitedEvents.end(), Handle) !=
         WaitedEvents.end();
}

// A best-effort check that nothing held in the runtime reached the backend: no
// kernel was launched and no barrier with a wait list was issued within a
// short interval. A pass does not prove the work is held; a failure proves it
// was not.
bool nothingReachedBackend() {
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  return kernelLaunches() == 0 && barriersWithWaitList().empty();
}

// The backend events which report UR_EVENT_STATUS_QUEUED: in the backend, but
// not yet submitted to the device. A consumer on another queue flushes the
// queue of a signal in that state. Guarded by BackendMutex.
std::set<ur_event_handle_t> QueuedEvents;

ur_result_t redefinedUrEventGetInfoQueued(void *pParams) {
  redefinedUrEventGetInfo(pParams);
  auto params = *static_cast<ur_event_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_EVENT_INFO_COMMAND_EXECUTION_STATUS &&
      *params.ppPropValue) {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    if (QueuedEvents.count(*params.phEvent))
      *static_cast<ur_event_status_t *>(*params.ppPropValue) =
          UR_EVENT_STATUS_QUEUED;
  }
  return UR_RESULT_SUCCESS;
}

// Lets the test report backend events as queued for as long as it lives.
struct QueuedStatusScope {
  QueuedStatusScope() {
    {
      std::lock_guard<std::mutex> Lock(BackendMutex);
      QueuedEvents.clear();
    }
    mock::getCallbacks().set_replace_callback("urEventGetInfo",
                                              &redefinedUrEventGetInfoQueued);
  }
  ~QueuedStatusScope() {
    mock::getCallbacks().set_replace_callback("urEventGetInfo",
                                              &redefinedUrEventGetInfo);
    std::lock_guard<std::mutex> Lock(BackendMutex);
    QueuedEvents.clear();
  }
  void add(ur_event_handle_t Handle) {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    QueuedEvents.insert(Handle);
  }
  void remove(ur_event_handle_t Handle) {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    QueuedEvents.erase(Handle);
  }
};

// The wait lists of the asynchronous allocations. Guarded by BackendMutex.
std::vector<std::vector<ur_event_handle_t>> AllocWaitLists;
alignas(64) unsigned char AllocStorage[64];

// The default mock gives an asynchronous allocation no memory.
ur_result_t redefinedUrEnqueueUSMDeviceAllocExp(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_usm_device_alloc_exp_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(BackendMutex);
  AllocWaitLists.push_back(
      recordWaitList(*params.pphEventWaitList, *params.pnumEventsInWaitList));
  **params.pppMem = AllocStorage;
  if (*params.pphEvent) {
    ur_event_handle_t Handle = newFakeEvent();
    producedBy(Handle, *params.phQueue, /*StaysPending=*/false);
    **params.pphEvent = Handle;
  }
  return UR_RESULT_SUCCESS;
}

// The device each backend event of a signal was created for, in the order of
// CreatedEvents. Guarded by BackendMutex.
std::vector<ur_device_handle_t> CreatedEventDevices;

ur_result_t redefinedUrEventCreateExpRecordingDevice(void *pParams) {
  auto params = *static_cast<ur_event_create_exp_params_t *>(pParams);
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    CreatedEventDevices.push_back(*params.phDevice);
  }
  return redefinedUrEventCreateExp(pParams);
}

// The backend events whose profiling information was queried. Each of them
// reports its handle as its timestamps. Guarded by BackendMutex.
std::vector<ur_event_handle_t> ProfiledEvents;

ur_result_t redefinedUrEventGetProfilingInfo(void *pParams) {
  auto params = *static_cast<ur_event_get_profiling_info_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(BackendMutex);
  ProfiledEvents.push_back(*params.phEvent);
  if (*params.ppPropValue && *params.ppropSize >= sizeof(uint64_t))
    *static_cast<uint64_t *>(*params.ppPropValue) =
        reinterpret_cast<std::uintptr_t>(*params.phEvent);
  if (*params.ppPropSizeRet)
    **params.ppPropSizeRet = sizeof(uint64_t);
  return UR_RESULT_SUCCESS;
}

// A platform with two devices.
ur_result_t redefinedUrDeviceGetTwoDevices(void *pParams) {
  auto params = *static_cast<ur_device_get_params_t *>(pParams);
  if (*params.ppNumDevices)
    **params.ppNumDevices = 2;
  if (*params.pphDevices && *params.pNumEntries >= 2) {
    (*params.pphDevices)[0] = reinterpret_cast<ur_device_handle_t>(1);
    (*params.pphDevices)[1] = reinterpret_cast<ur_device_handle_t>(2);
  }
  return UR_RESULT_SUCCESS;
}

ur_device_handle_t deviceHandleOf(const sycl::device &D) {
  return sycl::detail::getSyclObjImpl(D)->getHandleRef();
}

// tests-10-02 U08: mixed wait vectors, duplicates and the empty vector.

// tests-10-02 U08. A wait vector with a pending signal, a complete one, a
// pending signal in another context and a repeated entry, held behind a host
// task, then one of its events signaled again. The wait waits for the
// signals it was given: the one in the other context is bridged, the others
// reach the barrier, the later signal does not. The mock does not model the
// backend ordering, so the oracle is the barrier's wait list, not the queue's
// completion.
TEST_F(ReusableEventsQueueTest, MixedWaitVectorWaitsForEveryInput) {
  sycl::context Ctx2{Dev};
  sycl::queue SQ = inOrderQueue();
  sycl::queue CQ = inOrderQueue();
  sycl::queue RQ = inOrderQueue();
  sycl::queue XQ = inOrderQueue(Ctx2);
  sycl::queue Q = inOrderQueue();
  CompleteAllAtScopeExit CompleteAll;
  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};

  sycl::event P = syclex::make_event(Ctx);
  signalPending(SQ, P);
  const ur_event_handle_t PH = handleOf(P);
  sycl::event C = syclex::make_event(Ctx);
  syclex::enqueue_signal_event(CQ, C);
  const ur_event_handle_t CH = handleOf(C);
  ASSERT_TRUE(isComplete(C));
  sycl::event X = syclex::make_event(Ctx2);
  signalPending(XQ, X);
  const ur_event_handle_t XH = handleOf(X);
  ASSERT_NE(PH, nullptr);
  ASSERT_NE(CH, nullptr);
  ASSERT_NE(XH, nullptr);

  blockQueue(Q, Gate);
  ASSERT_TRUE(Gate->waitEntered());
  syclex::enqueue_wait_events(Q, {P, C, X, P});
  // A later signal of P, which the wait must not pick up.
  signalPending(RQ, P);
  const ur_event_handle_t PH2 = handleOf(P);
  ASSERT_NE(PH2, PH);
  submitKernel(Q);

  Gate->open();
  // The signal in the other context is waited for on the host.
  EXPECT_TRUE(eventually([&] { return waitedForLocked(XH); }));
  // Best effort: the barrier and the kernel behind it are held.
  EXPECT_TRUE(nothingReachedBackend());
  complete(PH2);
  EXPECT_TRUE(nothingReachedBackend());

  complete(XH);
  ASSERT_TRUE(eventually([&] { return KernelLaunchWaitLists.size() == 1; }));
  {
    std::vector<std::vector<ur_event_handle_t>> Barriers =
        barriersWithWaitList();
    ASSERT_EQ(Barriers.size(), 1u);
    const std::vector<ur_event_handle_t> &WaitList = Barriers[0];
    EXPECT_NE(std::find(WaitList.begin(), WaitList.end(), PH), WaitList.end());
    for (ur_event_handle_t Handle : WaitList) {
      EXPECT_TRUE(Handle == PH || Handle == CH)
          << "unexpected handle " << static_cast<void *>(Handle)
          << " in the wait list";
    }
  }
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    for (ur_event_handle_t Handle : KernelLaunchWaitLists[0]) {
      EXPECT_NE(Handle, XH);
      EXPECT_NE(Handle, PH2);
      EXPECT_TRUE(KnownHandles.count(Handle) != 0)
          << "foreign handle " << static_cast<void *>(Handle);
    }
  }

  complete(PH);
  Q.wait();
}

// tests-10-02 U08. The same with an ordinary host-task event in the vector:
// the barrier waits for the host task too.
// Known defect: review-10-02 #5 (the public waits reject every host event,
// host-task events included).
TEST_F(ReusableEventsQueueTest, DISABLED_MixedWaitVectorWithHostTaskEvent) {
  sycl::context Ctx2{Dev};
  sycl::queue SQ = inOrderQueue();
  sycl::queue CQ = inOrderQueue();
  sycl::queue RQ = inOrderQueue();
  sycl::queue HQ = inOrderQueue();
  sycl::queue XQ = inOrderQueue(Ctx2);
  sycl::queue Q = inOrderQueue();
  CompleteAllAtScopeExit CompleteAll;
  auto Gate = std::make_shared<HostTaskGate>();
  auto HostGate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGates[2] = {{Gate}, {HostGate}};

  sycl::event P = syclex::make_event(Ctx);
  signalPending(SQ, P);
  const ur_event_handle_t PH = handleOf(P);
  sycl::event C = syclex::make_event(Ctx);
  syclex::enqueue_signal_event(CQ, C);
  const ur_event_handle_t CH = handleOf(C);
  sycl::event X = syclex::make_event(Ctx2);
  signalPending(XQ, X);
  const ur_event_handle_t XH = handleOf(X);
  const sycl::event H = blockQueue(HQ, HostGate);
  ASSERT_TRUE(HostGate->waitEntered());

  blockQueue(Q, Gate);
  ASSERT_TRUE(Gate->waitEntered());
  ASSERT_NO_THROW(syclex::enqueue_wait_events(Q, {P, C, H, X, P}));
  signalPending(RQ, P);
  const ur_event_handle_t PH2 = handleOf(P);
  ASSERT_NE(PH2, PH);
  submitKernel(Q);

  Gate->open();
  EXPECT_TRUE(eventually([&] { return waitedForLocked(XH); }));
  complete(PH2);
  complete(XH);
  // Best effort: the host task still holds the barrier.
  EXPECT_TRUE(nothingReachedBackend());

  HostGate->open();
  ASSERT_TRUE(eventually([&] { return KernelLaunchWaitLists.size() == 1; }));
  EXPECT_TRUE(isComplete(H));
  {
    std::vector<std::vector<ur_event_handle_t>> Barriers =
        barriersWithWaitList();
    ASSERT_EQ(Barriers.size(), 1u);
    const std::vector<ur_event_handle_t> &WaitList = Barriers[0];
    EXPECT_NE(std::find(WaitList.begin(), WaitList.end(), PH), WaitList.end());
    for (ur_event_handle_t Handle : WaitList) {
      EXPECT_TRUE(Handle == PH || Handle == CH)
          << "unexpected handle " << static_cast<void *>(Handle)
          << " in the wait list";
    }
  }

  complete(PH);
  Q.wait();
}

// tests-10-02 U08. An empty wait vector on an out-of-order queue is a no-op:
// it issues no barrier, and later work does not wait for the work submitted
// before it, neither a kernel held in the runtime nor one pending in the
// backend.
TEST_F(ReusableEventsQueueTest, EmptyWaitVectorIsNoOp) {
  sycl::queue Q{Ctx, Dev};
  CompleteAllAtScopeExit CompleteAll;
  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};

  const sycl::event HostTask = blockQueue(Q, Gate);
  submitKernelAfter(Q, HostTask);
  setKernelsStayPending(true);
  submitKernel(Q);
  setKernelsStayPending(false);
  ASSERT_EQ(kernelLaunches(), 1u);

  size_t Barriers = 0;
  size_t BarrierOuts = 0;
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    Barriers = BarrierWaitLists.size();
    BarrierOuts = BarrierOutEvents.size();
  }
  syclex::enqueue_wait_events(Q, {});
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    EXPECT_EQ(BarrierWaitLists.size(), Barriers);
    EXPECT_EQ(BarrierOutEvents.size(), BarrierOuts);
  }

  // Independent work runs while the earlier kernels are held.
  submitKernel(Q);
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    ASSERT_EQ(KernelLaunchWaitLists.size(), 2u);
    EXPECT_TRUE(KernelLaunchWaitLists[1].empty());
  }

  Gate->open();
  completeAll();
  Q.wait();
  EXPECT_EQ(kernelLaunches(), 3u);
}

// tests-10-02 U09: the public waits accept ordinary host-task events.

// tests-10-02 U09. Both public waits for a host task held at a gate are
// accepted, and the device work submitted after them waits for the host task.
// Known defect: review-10-02 #5 (the public waits reject every host event,
// host-task events included).
TEST_F(ReusableEventsQueueTest, DISABLED_HostTaskEventWaitsAreAccepted) {
  sycl::queue HQ = inOrderQueue();
  sycl::queue Q1 = inOrderQueue();
  sycl::queue Q2 = inOrderQueue();
  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};

  const sycl::event H = blockQueue(HQ, Gate);
  ASSERT_TRUE(Gate->waitEntered());
  EXPECT_NO_THROW(syclex::enqueue_wait_event(Q1, H));
  submitKernel(Q1);
  EXPECT_NO_THROW(syclex::enqueue_wait_events(Q2, {H}));
  submitKernel(Q2);
  EXPECT_EQ(kernelLaunches(), 0u);
  EXPECT_FALSE(isComplete(H));

  Gate->open();
  Q1.wait();
  Q2.wait();
  EXPECT_TRUE(isComplete(H));
  EXPECT_EQ(kernelLaunches(), 2u);
}

// tests-10-02 U09. The same with a host task which is complete already.
// Known defect: review-10-02 #5 (the public waits reject every host event,
// host-task events included).
TEST_F(ReusableEventsQueueTest, DISABLED_CompletedHostTaskEventWaitIsAccepted) {
  sycl::queue HQ = inOrderQueue();
  sycl::queue Q1 = inOrderQueue();
  sycl::queue Q2 = inOrderQueue();

  sycl::event H = HQ.submit([&](sycl::handler &CGH) { CGH.host_task([] {}); });
  H.wait();
  EXPECT_NO_THROW(syclex::enqueue_wait_event(Q1, H));
  submitKernel(Q1);
  EXPECT_NO_THROW(syclex::enqueue_wait_events(Q2, {H}));
  submitKernel(Q2);
  Q1.wait();
  Q2.wait();
  EXPECT_EQ(kernelLaunches(), 2u);
}

// tests-10-02 U09. The same with a host task held in another context.
// Known defect: review-10-02 #5 (the public waits reject every host event,
// host-task events included).
TEST_F(ReusableEventsQueueTest,
       DISABLED_OtherContextHostTaskEventWaitIsAccepted) {
  sycl::context Ctx2{Dev};
  sycl::queue HQ = inOrderQueue(Ctx2);
  sycl::queue Q1 = inOrderQueue();
  sycl::queue Q2 = inOrderQueue();
  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};

  const sycl::event H = blockQueue(HQ, Gate);
  ASSERT_TRUE(Gate->waitEntered());
  EXPECT_NO_THROW(syclex::enqueue_wait_event(Q1, H));
  submitKernel(Q1);
  EXPECT_NO_THROW(syclex::enqueue_wait_events(Q2, {H}));
  submitKernel(Q2);
  EXPECT_EQ(kernelLaunches(), 0u);

  Gate->open();
  Q1.wait();
  Q2.wait();
  EXPECT_TRUE(isComplete(H));
  EXPECT_EQ(kernelLaunches(), 2u);
}

// tests-10-02 U13: the last-event query and the asynchronous allocation built
// on it.

// tests-10-02 U13, control.
TEST_F(ReusableEventsQueueTest, LastEventOfEmptyInOrderQueueIsNullopt) {
  sycl::queue Q = inOrderQueue();
  EXPECT_FALSE(Q.ext_oneapi_get_last_event().has_value());
}

// tests-10-02 U13, control.
TEST_F(ReusableEventsQueueTest, LastEventOnOutOfOrderQueueThrows) {
  sycl::queue Q{Ctx, Dev};
  try {
    std::ignore = Q.ext_oneapi_get_last_event();
    FAIL() << "the last event of an out-of-order queue did not throw";
  } catch (const sycl::exception &Ex) {
    EXPECT_EQ(Ex.code(), sycl::make_error_code(sycl::errc::invalid));
  }
}

// tests-10-02 U13, control: the external event set after the last command is
// the last event, a copy of it, until a command consumes it.
TEST_F(ReusableEventsQueueTest, LastEventAfterExternalEventIsItsCopy) {
  sycl::queue SQ = inOrderQueue();
  sycl::queue Q = inOrderQueue();
  submitKernel(Q);
  sycl::event X = syclex::make_event(Ctx);
  syclex::enqueue_signal_event(SQ, X);
  Q.ext_oneapi_set_external_event(X);

  std::optional<sycl::event> Last = Q.ext_oneapi_get_last_event();
  ASSERT_TRUE(Last.has_value());
  EXPECT_EQ(*Last, X);
  EXPECT_EQ(handleOf(*Last), handleOf(X));

  submitKernel(Q);
  std::optional<sycl::event> After = Q.ext_oneapi_get_last_event();
  ASSERT_TRUE(After.has_value());
  EXPECT_NE(*After, X);
  Q.wait();
}

// tests-10-02 U13, control: a kernel which reached the backend is not kept as
// the last event of the queue; the query marks the end of the queue instead,
// whatever its event represents by then. The binding held here stands in for
// a consumer of the kernel's signal, so the event moves on to a new one.
TEST_F(ReusableEventsQueueTest, LastEventOfEnqueuedKernelIsQueueMarker) {
  sycl::queue Q1 = inOrderQueue();
  sycl::queue Q2 = inOrderQueue();
  CompleteAllAtScopeExit CompleteAll;

  setKernelsStayPending(true);
  sycl::event E = submitKernel(Q1);
  setKernelsStayPending(false);
  const ur_event_handle_t K = handleOf(E);
  ASSERT_NE(K, nullptr);
  std::shared_ptr<sycl::detail::event_binding> Held = bindingOf(E);
  syclex::enqueue_signal_event(Q2, E);
  const ur_event_handle_t H2 = handleOf(E);
  ASSERT_NE(H2, K);

  std::optional<sycl::event> Last = Q1.ext_oneapi_get_last_event();
  ASSERT_TRUE(Last.has_value());
  EXPECT_NE(*Last, E);
  const ur_event_handle_t Marker = handleOf(*Last);
  EXPECT_NE(Marker, nullptr);
  EXPECT_NE(Marker, H2);
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    EXPECT_EQ(HandleQueues[Marker], handleOf(Q1));
  }

  complete(K);
  Q1.wait();
}

// Submits kernel E to Q1 held behind a host task on another queue, so that Q1
// keeps E as its last event, and signals E again on Q2 while it is held. Then
// lets the kernel reach the backend, where it stays pending, and returns its
// backend event.
ur_event_handle_t resignalHeldLastKernel(sycl::queue &Q1, sycl::queue &Q2,
                                         sycl::queue &HQ, sycl::event &E) {
  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};
  const sycl::event HostTask = blockQueue(HQ, Gate);
  E = submitKernelAfter(Q1, HostTask);
  EXPECT_EQ(handleOf(E), nullptr);
  const std::shared_ptr<sycl::detail::event_binding> Original = bindingOf(E);

  syclex::enqueue_signal_event(Q2, E);
  EXPECT_NE(bindingOf(E), Original);
  EXPECT_TRUE(isComplete(E));

  setKernelsStayPending(true);
  Gate->open();
  EXPECT_TRUE(eventually([&] { return KernelLaunchWaitLists.size() == 1; }));
  setKernelsStayPending(false);
  std::lock_guard<std::mutex> Lock(BackendMutex);
  return KernelEvents.empty() ? nullptr : KernelEvents[0];
}

// tests-10-02 U13. The last event of Q1 is its last command, the kernel, not
// the later signal of the kernel's event on Q2.
// Known defect: review-10-02 #7 (getLastEvent hands out the event of the last
// command, which represents its latest signal, not the captured one).
TEST_F(ReusableEventsQueueTest, DISABLED_LastEventTracksQueueAfterResignal) {
  sycl::queue HQ = inOrderQueue();
  sycl::queue Q1 = inOrderQueue();
  sycl::queue Q2 = inOrderQueue();
  sycl::queue Q3 = inOrderQueue();
  CompleteAllAtScopeExit CompleteAll;
  sycl::event E;
  const ur_event_handle_t K = resignalHeldLastKernel(Q1, Q2, HQ, E);
  ASSERT_NE(K, nullptr);

  std::optional<sycl::event> Last = Q1.ext_oneapi_get_last_event();
  ASSERT_TRUE(Last.has_value());
  EXPECT_FALSE(isComplete(*Last));
  EXPECT_EQ(handleOf(*Last), K);

  submitKernelAfter(Q3, *Last);
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    ASSERT_EQ(KernelLaunchWaitLists.size(), 2u);
    EXPECT_EQ(KernelLaunchWaitLists[1], std::vector<ur_event_handle_t>{K});
  }

  complete(K);
  Q1.wait();
  Q3.wait();
}

// tests-10-02 U13. An asynchronous allocation on Q1 follows the last command
// of Q1, which it finds through the last-event query.
// Known defect: review-10-02 #7 (getLastEvent hands out the event of the last
// command, which represents its latest signal, not the captured one).
TEST_F(ReusableEventsQueueTest,
       DISABLED_AsyncMallocOrdersAfterQueueAfterResignal) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueUSMDeviceAllocExp", &redefinedUrEnqueueUSMDeviceAllocExp);
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    AllocWaitLists.clear();
  }
  sycl::queue HQ = inOrderQueue();
  sycl::queue Q1 = inOrderQueue();
  sycl::queue Q2 = inOrderQueue();
  CompleteAllAtScopeExit CompleteAll;
  sycl::event E;
  const ur_event_handle_t K = resignalHeldLastKernel(Q1, Q2, HQ, E);
  ASSERT_NE(K, nullptr);
  const ur_event_handle_t H2 = handleOf(E);

  void *Ptr = syclex::async_malloc(Q1, sycl::usm::alloc::device, 64);
  EXPECT_EQ(Ptr, static_cast<void *>(AllocStorage));
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    ASSERT_EQ(AllocWaitLists.size(), 1u);
    EXPECT_EQ(AllocWaitLists[0], std::vector<ur_event_handle_t>{K});
    EXPECT_EQ(std::find(AllocWaitLists[0].begin(), AllocWaitLists[0].end(), H2),
              AllocWaitLists[0].end());
  }

  complete(K);
  syclex::async_free(Q1, Ptr);
  Q1.wait();
}

// tests-10-02 U14: external events are captured when they are consumed.

// tests-10-02 U14. The command consuming the external event keeps the signal
// it consumed, held in the runtime, when the event is signaled again.
TEST_F(ReusableEventsQueueTest, ExternalEventCapturedWhenConsumed) {
  sycl::queue SQ = inOrderQueue();
  sycl::queue SQ2 = inOrderQueue();
  sycl::queue Q = inOrderQueue();
  CompleteAllAtScopeExit CompleteAll;
  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};

  blockQueue(SQ, Gate);
  sycl::event E = syclex::make_event(Ctx);
  syclex::enqueue_signal_event(SQ, E);
  const std::shared_ptr<sycl::detail::event_binding> S1 = bindingOf(E);
  ASSERT_EQ(S1->getHandle(), nullptr);

  Q.ext_oneapi_set_external_event(E);
  submitKernel(Q);
  signalPending(SQ2, E);
  const ur_event_handle_t H2 = handleOf(E);
  ASSERT_NE(bindingOf(E), S1);
  EXPECT_EQ(kernelLaunches(), 0u);

  Gate->open();
  ASSERT_TRUE(eventually([&] { return KernelLaunchWaitLists.size() == 1; }));
  const ur_event_handle_t H1 = S1->getHandle();
  ASSERT_NE(H1, nullptr);
  EXPECT_NE(H1, H2);
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    EXPECT_EQ(KernelLaunchWaitLists[0], std::vector<ur_event_handle_t>{H1});
  }

  complete(H2);
  Q.wait();
  SQ.wait();
}

// tests-10-02 U14. An external event signaled again before the next command
// is consumed with the signal it represents then. The binding held here
// stands in for a consumer of the first signal, so the event moves on to a
// new one.
TEST_F(ReusableEventsQueueTest, ExternalEventFollowsSignalAtConsumption) {
  sycl::queue SQ = inOrderQueue();
  sycl::queue SQ2 = inOrderQueue();
  sycl::queue Q = inOrderQueue();
  CompleteAllAtScopeExit CompleteAll;

  sycl::event E = syclex::make_event(Ctx);
  signalPending(SQ, E);
  const ur_event_handle_t H1 = handleOf(E);
  std::shared_ptr<sycl::detail::event_binding> Held = bindingOf(E);
  Q.ext_oneapi_set_external_event(E);
  signalPending(SQ2, E);
  const ur_event_handle_t H2 = handleOf(E);
  ASSERT_NE(H2, H1);

  submitKernel(Q);
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    ASSERT_EQ(KernelLaunchWaitLists.size(), 1u);
    EXPECT_EQ(KernelLaunchWaitLists[0], std::vector<ur_event_handle_t>{H2});
  }

  complete(H1);
  complete(H2);
  Q.wait();
}

// tests-10-02 U14. Only the external event set last affects the next command,
// and only that one.
TEST_F(ReusableEventsQueueTest, ReplacedExternalEventOnlyLastCounts) {
  sycl::queue SQ1 = inOrderQueue();
  sycl::queue SQ2 = inOrderQueue();
  sycl::queue Q = inOrderQueue();
  CompleteAllAtScopeExit CompleteAll;

  sycl::event E1 = syclex::make_event(Ctx);
  signalPending(SQ1, E1);
  sycl::event E2 = syclex::make_event(Ctx);
  signalPending(SQ2, E2);
  const ur_event_handle_t H2 = handleOf(E2);

  Q.ext_oneapi_set_external_event(E1);
  Q.ext_oneapi_set_external_event(E2);
  submitKernel(Q);
  submitKernel(Q);
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    ASSERT_EQ(KernelLaunchWaitLists.size(), 2u);
    EXPECT_EQ(KernelLaunchWaitLists[0], std::vector<ur_event_handle_t>{H2});
    EXPECT_TRUE(KernelLaunchWaitLists[1].empty());
  }

  completeAll();
  Q.wait();
}

// tests-10-02 U14. A queue wait before the next command waits for the
// external event and clears it: a later signal of the event does not hold the
// next command.
TEST_F(ReusableEventsQueueTest, QueueWaitClearsExternalEvent) {
  sycl::queue SQ = inOrderQueue();
  sycl::queue Q = inOrderQueue();
  std::future<void> Waiter;
  CompleteAllAtScopeExit CompleteAll;

  sycl::event E = syclex::make_event(Ctx);
  signalPending(SQ, E);
  const ur_event_handle_t H = handleOf(E);
  Q.ext_oneapi_set_external_event(E);

  Waiter = std::async(std::launch::async, [&] { Q.wait(); });
  EXPECT_TRUE(eventually([&] { return waitedForLocked(H); }));
  EXPECT_TRUE(stillBlocked(Waiter));
  complete(H);
  ASSERT_TRUE(finishes(Waiter));

  // The wait is over before the event is signaled again.
  signalPending(SQ, E);
  submitKernel(Q);
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    ASSERT_EQ(KernelLaunchWaitLists.size(), 1u);
    EXPECT_TRUE(KernelLaunchWaitLists[0].empty());
  }

  completeAll();
  Q.wait();
}

// tests-10-02 U24: flushes and in-order filtering use the queue of the
// captured signal.

// tests-10-02 U24. Two consumers on Q3 capture kernel E's signal on Q1, then E
// is signaled again on Q2. The consumers flush Q1, the queue of the signal
// they wait for, once: the second one finds it flushed.
TEST_F(ReusableEventsQueueTest, FlushFollowsCapturedSignalQueue) {
  sycl::queue Q1 = inOrderQueue();
  sycl::queue Q2 = inOrderQueue();
  sycl::queue Q3 = inOrderQueue();
  QueuedStatusScope Queued;
  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};

  sycl::event E = submitKernel(Q1);
  const ur_event_handle_t K = handleOf(E);
  ASSERT_NE(K, nullptr);
  Queued.add(K);
  blockQueue(Q3, Gate);
  submitKernelAfter(Q3, E);
  submitKernelAfter(Q3, E);

  syclex::enqueue_signal_event(Q2, E);
  const ur_event_handle_t H2 = handleOf(E);
  ASSERT_NE(H2, K);
  Queued.add(H2);
  EXPECT_EQ(flushesOf(handleOf(Q1)), 0u);

  Gate->open();
  ASSERT_TRUE(eventually([&] { return KernelLaunchWaitLists.size() == 3; }));
  EXPECT_EQ(flushesOf(handleOf(Q1)), 1u);
  EXPECT_EQ(flushesOf(handleOf(Q2)), 0u);
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    EXPECT_EQ(KernelLaunchWaitLists[1], std::vector<ur_event_handle_t>{K});
    EXPECT_EQ(KernelLaunchWaitLists[2], std::vector<ur_event_handle_t>{K});
  }
  Q3.wait();
}

// tests-10-02 U24. A consumer on the queue of the signal itself does not
// flush it.
TEST_F(ReusableEventsQueueTest, SameQueueConsumerDoesNotFlush) {
  sycl::queue Q1 = inOrderQueue();
  QueuedStatusScope Queued;
  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};

  sycl::event E = submitKernel(Q1);
  const ur_event_handle_t K = handleOf(E);
  ASSERT_NE(K, nullptr);
  Queued.add(K);
  blockQueue(Q1, Gate);
  submitKernelAfter(Q1, E);

  Gate->open();
  ASSERT_TRUE(eventually([&] { return KernelLaunchWaitLists.size() == 2; }));
  EXPECT_EQ(flushesOf(handleOf(Q1)), 0u);
  Q1.wait();
}

// tests-10-02 U24. A consumer of a signal whose queue is gone does not flush
// it: releasing the queue flushed it.
TEST_F(ReusableEventsQueueTest, ReleasedSignalQueueIsNotFlushed) {
  sycl::queue Q3 = inOrderQueue();
  QueuedStatusScope Queued;
  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};

  blockQueue(Q3, Gate);
  sycl::event E;
  ur_event_handle_t K = nullptr;
  {
    sycl::queue Q1 = inOrderQueue();
    E = submitKernel(Q1);
    K = handleOf(E);
    ASSERT_NE(K, nullptr);
    Queued.add(K);
    submitKernelAfter(Q3, E);
  }
  ASSERT_TRUE(bindingOf(E)->MQueue.expired());
  size_t Flushes = 0;
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    Flushes = FlushedQueues.size();
  }

  Gate->open();
  ASSERT_TRUE(eventually([&] { return KernelLaunchWaitLists.size() == 2; }));
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    EXPECT_EQ(FlushedQueues.size(), Flushes);
    EXPECT_EQ(KernelLaunchWaitLists[1], std::vector<ur_event_handle_t>{K});
  }
  EXPECT_TRUE(bindingOf(E)->MIsFlushed);
  Q3.wait();
}

// tests-10-02 U24. A binding reused for a new signal is flushed again by the
// consumers of the new signal.
TEST_F(ReusableEventsQueueTest, ReusedBindingFlushesAgain) {
  // Out-of-order queues keep no bookkeeping of their last signal, so nothing
  // but the event refers to a signal once it is done.
  sycl::queue Q1{Ctx, Dev};
  QueuedStatusScope Queued;
  sycl::event E = syclex::make_event(Ctx);
  auto Impl = sycl::detail::getSyclObjImpl(E);
  // Only scheduler commands flush; the buffer takes the consumer there. The
  // consumer and its queue are gone before the binding is reused.
  auto Consume = [&] {
    sycl::queue Consumers = inOrderQueue();
    sycl::buffer<int, 1> Buf{1};
    Consumers.submit([&](sycl::handler &CGH) {
      sycl::accessor A{Buf, CGH, sycl::write_only};
      CGH.depends_on(E);
      CGH.single_task<BindingTestKernel>([]() {});
    });
    Consumers.wait();
  };

  syclex::enqueue_signal_event(Q1, E);
  sycl::detail::event_binding *Binding = Impl->getBinding().get();
  const ur_event_handle_t H1 = handleOf(E);
  ASSERT_NE(H1, nullptr);
  Queued.add(H1);
  Consume();
  EXPECT_TRUE(Binding->MIsFlushed);
  EXPECT_EQ(flushesOf(handleOf(Q1)), 1u);

  // The first signal is done.
  Queued.remove(H1);
  Q1.wait();
  ASSERT_TRUE(eventually([&] { return Impl->getBinding().use_count() == 1; }));
  syclex::enqueue_signal_event(Q1, E);
  ASSERT_EQ(Impl->getBinding().get(), Binding)
      << "the signal did not reuse the binding";
  EXPECT_FALSE(Binding->MIsFlushed);
  const ur_event_handle_t H2 = handleOf(E);
  ASSERT_NE(H2, nullptr);
  Queued.add(H2);
  Consume();
  EXPECT_TRUE(Binding->MIsFlushed);
  EXPECT_EQ(flushesOf(handleOf(Q1)), 2u);
  Q1.wait();
}

// tests-10-02 U24. A consumer on Q1 of kernel E's signal on Q2 keeps it in its
// wait list when E is signaled again on Q1: the in-order filtering looks at
// the queue of the captured signal, not at the event's latest queue.
TEST_F(ReusableEventsQueueTest, InOrderFilteringUsesCapturedWorkerQueue) {
  sycl::queue Q2 = inOrderQueue();
  sycl::queue Q1 = inOrderQueue();
  CompleteAllAtScopeExit CompleteAll;
  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};

  setKernelsStayPending(true);
  sycl::event E = submitKernel(Q2);
  setKernelsStayPending(false);
  const ur_event_handle_t K = handleOf(E);
  ASSERT_NE(K, nullptr);

  blockQueue(Q1, Gate);
  submitKernelAfter(Q1, E);
  syclex::enqueue_signal_event(Q1, E);

  Gate->open();
  ASSERT_TRUE(eventually([&] { return KernelLaunchWaitLists.size() == 2; }));
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    const std::vector<ur_event_handle_t> &WaitList = KernelLaunchWaitLists[1];
    EXPECT_NE(std::find(WaitList.begin(), WaitList.end(), K), WaitList.end());
  }

  complete(K);
  Q1.wait();
  Q2.wait();
}

// tests-10-02 U44: the device of a signal is the device of its queue.

class ReusableEventsDeviceAttributionTest
    : public ReusableEventsTest,
      public ::testing::WithParamInterface<bool> {};

INSTANTIATE_TEST_SUITE_P(ReusableEventsQueue,
                         ReusableEventsDeviceAttributionTest, ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool> &Info) {
                           return Info.param ? "OlderFirst" : "NewerFirst";
                         });

// tests-10-02 U44. In a context of two devices, an event is signaled on a
// queue of D1, captured, then signaled on a queue of D2 behind a host task.
// Each backend event is created for the device of the queue of its signal and
// enqueued there, the captured signal stays with D1, the signals complete
// independently in either order, and the profiling information is that of the
// latest signal. MSubmittedDevice has no reader or writer; nothing here
// depends on it.
TEST_P(ReusableEventsDeviceAttributionTest, SignalDeviceFollowsProducingQueue) {
  const bool OlderFirst = GetParam();
  mock::getCallbacks().set_replace_callback("urDeviceGet",
                                            &redefinedUrDeviceGetTwoDevices);
  mock::getCallbacks().set_replace_callback(
      "urEventCreateExp", &redefinedUrEventCreateExpRecordingDevice);
  mock::getCallbacks().set_replace_callback("urEventGetProfilingInfo",
                                            &redefinedUrEventGetProfilingInfo);
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    CreatedEventDevices.clear();
    ProfiledEvents.clear();
  }
  std::vector<sycl::device> Devices = Dev.get_platform().get_devices();
  if (Devices.size() < 2 || Devices[0] == Devices[1])
    GTEST_SKIP() << "the mock platform has no two distinct devices";
  const sycl::device D1 = Devices[0];
  const sycl::device D2 = Devices[1];
  sycl::context C12{std::vector<sycl::device>{D1, D2}};
  sycl::queue Q1{C12, D1, sycl::property::queue::in_order{}};
  sycl::queue Q2{C12, D2, sycl::property::queue::in_order{}};
  sycl::queue Q3{C12, D1, sycl::property::queue::in_order{}};
  CompleteAllAtScopeExit CompleteAll;
  auto Gate2 = std::make_shared<HostTaskGate>();
  auto Gate3 = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGates[2] = {{Gate2}, {Gate3}};

  sycl::event E = syclex::make_event(C12, syclex::enable_profiling{true});
  signalPending(Q1, E);
  const std::shared_ptr<sycl::detail::event_binding> S1 = bindingOf(E);
  const ur_event_handle_t H1 = handleOf(E);
  ASSERT_NE(H1, nullptr);
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    ASSERT_EQ(CreatedEvents.size(), 1u);
    EXPECT_EQ(CreatedEvents[0], H1);
    EXPECT_EQ(CreatedEventDevices[0], deviceHandleOf(D1));
    EXPECT_EQ(HandleQueues[H1], handleOf(Q1));
  }

  blockQueue(Q3, Gate3);
  submitKernelAfter(Q3, E);
  blockQueue(Q2, Gate2);
  syclex::enqueue_signal_event(Q2, E);
  ASSERT_NE(bindingOf(E), S1);

  setBarriersStayPending(true);
  Gate2->open();
  ASSERT_TRUE(eventually([&] { return CreatedEvents.size() == 2; }));
  const ur_event_handle_t H2 = handleOf(E);
  ASSERT_TRUE(eventually([&] { return HandleQueues.count(H2) != 0; }));
  setBarriersStayPending(false);
  ASSERT_NE(H2, nullptr);
  ASSERT_NE(H2, H1);
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    EXPECT_EQ(CreatedEvents[1], H2);
    EXPECT_EQ(CreatedEventDevices[1], deviceHandleOf(D2));
    EXPECT_EQ(HandleQueues[H2], handleOf(Q2));
    EXPECT_EQ(HandleQueues[H1], handleOf(Q1));
    for (ur_exp_event_flags_t Flags : CreatedEventFlags) {
      EXPECT_TRUE(Flags & UR_EXP_EVENT_FLAG_ENABLE_PROFILING);
    }
  }

  Gate3->open();
  ASSERT_TRUE(eventually([&] { return KernelLaunchWaitLists.size() == 1; }));
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    EXPECT_EQ(KernelLaunchWaitLists[0], std::vector<ur_event_handle_t>{H1});
  }

  if (OlderFirst) {
    complete(H1);
    EXPECT_TRUE(S1->isCompleted());
    EXPECT_FALSE(isComplete(E));
    complete(H2);
    EXPECT_TRUE(isComplete(E));
  } else {
    complete(H2);
    EXPECT_TRUE(isComplete(E));
    EXPECT_FALSE(S1->isCompleted());
    complete(H1);
    EXPECT_TRUE(S1->isCompleted());
  }

  EXPECT_EQ(E.get_profiling_info<sycl::info::event_profiling::command_end>(),
            reinterpret_cast<std::uintptr_t>(H2));
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    ASSERT_FALSE(ProfiledEvents.empty());
    EXPECT_EQ(ProfiledEvents.back(), H2);
  }
  Q1.wait();
  Q2.wait();
  Q3.wait();
}

} // anonymous namespace
