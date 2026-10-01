//==-- ReusableEventsBinding.cpp --- One binding per signal of an event ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A command may stay inside the SYCL runtime after it has been submitted, and
// read its dependencies only when it is finally enqueued. The extension
// specifies that a dependency on a reusable event is captured when the command
// is submitted, so a later enqueue_signal_event must not change it. The runtime
// gets this by giving each signal an event_binding of its own: a dependency
// captures the binding, and enqueue_signal_event moves the event on to a new
// binding whenever something still refers to the previous one (and reuses the
// binding, backend event included, when nothing does).
//
// The tests below hold a command behind a host task, re-signal the event it
// depends on in the meantime, and check which backend event the command waits
// for once it reaches the backend.
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

#include <sycl/ext/oneapi/experimental/reusable_events.hpp>

#include <detail/event_impl.hpp>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

class BindingTestKernel;
MOCK_INTEGRATION_HEADER(BindingTestKernel)

namespace {

namespace syclex = sycl::ext::oneapi::experimental;

sycl::unittest::MockDeviceImage DevImage =
    sycl::unittest::generateDefaultImage({"BindingTestKernel"});
sycl::unittest::MockDeviceImageArray<1> DevImageArray = {&DevImage};

// Every backend event the runtime creates gets a distinct fake handle, so that
// the tests can tell the signals apart.
std::uintptr_t NextEventHandle = 0x1000;
std::vector<ur_event_handle_t> CreatedEvents;
std::map<ur_event_handle_t, int> RetainCounts;
std::map<ur_event_handle_t, int> ReleaseCounts;

// What reached the backend.
std::vector<std::vector<ur_event_handle_t>> KernelLaunchWaitLists;
std::vector<ur_event_handle_t> KernelEvents;
std::vector<std::vector<ur_event_handle_t>> BarrierWaitLists;
std::vector<std::vector<ur_event_handle_t>> CommandBufferWaitLists;
std::vector<ur_event_handle_t> CommandBufferEvents;
std::vector<ur_event_handle_t> WaitedEvents;
std::mutex BackendMutex;

ur_event_handle_t newFakeEvent() {
  return reinterpret_cast<ur_event_handle_t>(NextEventHandle++);
}

ur_result_t redefinedUrEventCreateExp(void *pParams) {
  auto params = *static_cast<ur_event_create_exp_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(BackendMutex);
  ur_event_handle_t Handle = newFakeEvent();
  **params.pphEvent = Handle;
  CreatedEvents.push_back(Handle);
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedUrEventRetain(void *pParams) {
  auto params = *static_cast<ur_event_retain_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(BackendMutex);
  RetainCounts[*params.phEvent]++;
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedUrEventRelease(void *pParams) {
  auto params = *static_cast<ur_event_release_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(BackendMutex);
  ReleaseCounts[*params.phEvent]++;
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedUrEventGetInfo(void *pParams) {
  auto params = *static_cast<ur_event_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_EVENT_INFO_COMMAND_EXECUTION_STATUS) {
    auto *Result = reinterpret_cast<ur_event_status_t *>(*params.ppPropValue);
    *Result = UR_EVENT_STATUS_COMPLETE;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedUrEventWait(void *pParams) {
  auto params = *static_cast<ur_event_wait_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(BackendMutex);
  WaitedEvents.insert(WaitedEvents.end(), *params.pphEventWaitList,
                      *params.pphEventWaitList + *params.pnumEvents);
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedUrEnqueueKernelLaunchWithArgsExp(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_kernel_launch_with_args_exp_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(BackendMutex);
  KernelLaunchWaitLists.emplace_back(*params.pphEventWaitList,
                                     *params.pphEventWaitList +
                                         *params.pnumEventsInWaitList);
  ur_event_handle_t Handle = newFakeEvent();
  KernelEvents.push_back(Handle);
  if (*params.pphEvent)
    **params.pphEvent = Handle;
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedUrEnqueueCommandBufferExp(void *pParams) {
  auto params = *static_cast<ur_enqueue_command_buffer_exp_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(BackendMutex);
  CommandBufferWaitLists.emplace_back(*params.pphEventWaitList,
                                      *params.pphEventWaitList +
                                          *params.pnumEventsInWaitList);
  ur_event_handle_t Handle = newFakeEvent();
  CommandBufferEvents.push_back(Handle);
  if (*params.pphEvent)
    **params.pphEvent = Handle;
  return UR_RESULT_SUCCESS;
}

// A barrier enqueued by the scheduler may express its dependencies through a
// plain events wait preceding the barrier call; record those too.
ur_result_t redefinedUrEnqueueEventsWait(void *pParams) {
  auto params = *static_cast<ur_enqueue_events_wait_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(BackendMutex);
  BarrierWaitLists.emplace_back(*params.pphEventWaitList,
                                *params.pphEventWaitList +
                                    *params.pnumEventsInWaitList);
  if (*params.pphEvent && **params.pphEvent == nullptr)
    **params.pphEvent = newFakeEvent();
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedUrEnqueueEventsWaitWithBarrierExt(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_events_wait_with_barrier_ext_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(BackendMutex);
  BarrierWaitLists.emplace_back(*params.pphEventWaitList,
                                *params.pphEventWaitList +
                                    *params.pnumEventsInWaitList);
  // A signal passes the reusable event in; any other barrier asks for a new
  // event.
  if (*params.pphEvent && **params.pphEvent == nullptr)
    **params.pphEvent = newFakeEvent();
  return UR_RESULT_SUCCESS;
}

ur_result_t after_urDeviceGetInfo(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_DEVICE_INFO_REUSABLE_EVENTS_SUPPORT_EXP) {
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(ur_bool_t);
    if (*params.ppPropValue)
      *static_cast<ur_bool_t *>(*params.ppPropValue) = ur_bool_t{true};
  }
  return UR_RESULT_SUCCESS;
}

// A gate for a host task to block on, so that everything submitted to an
// in-order queue after it stays inside the runtime until the gate is opened.
class HostTaskGate {
public:
  void wait() {
    std::unique_lock<std::mutex> Lock(MMutex);
    MCv.wait(Lock, [this] { return MReady; });
  }

  void open() {
    {
      std::lock_guard<std::mutex> Lock(MMutex);
      MReady = true;
    }
    MCv.notify_all();
  }

private:
  std::mutex MMutex;
  std::condition_variable MCv;
  bool MReady = false;
};

// Opens the gate when the scope is left, so that a failed assertion does not
// leave the host task - and the queue destructor waiting for it - blocked.
struct OpenAtScopeExit {
  std::shared_ptr<HostTaskGate> Gate;
  ~OpenAtScopeExit() { Gate->open(); }
};

// The host task shares ownership of the gate, so it stays valid for as long as
// the task may look at it.
sycl::event blockQueue(sycl::queue &Q, std::shared_ptr<HostTaskGate> Gate) {
  return Q.submit(
      [&Gate](sycl::handler &CGH) { CGH.host_task([Gate] { Gate->wait(); }); });
}

// The barrier (or the events wait preceding it) calls which had something to
// wait for, i.e. not the signals.
std::vector<std::vector<ur_event_handle_t>> barriersWithWaitList() {
  std::lock_guard<std::mutex> Lock(BackendMutex);
  std::vector<std::vector<ur_event_handle_t>> Result;
  for (const auto &WaitList : BarrierWaitLists)
    if (!WaitList.empty())
      Result.push_back(WaitList);
  return Result;
}

int releases(ur_event_handle_t Handle) {
  std::lock_guard<std::mutex> Lock(BackendMutex);
  return ReleaseCounts[Handle];
}

// Waits until \p Predicate holds, or a few seconds passed.
template <typename PredicateT> bool eventually(PredicateT Predicate) {
  const auto Deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
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

ur_event_handle_t handleOf(const sycl::event &E) {
  return sycl::detail::getSyclObjImpl(E)->getHandle();
}

class ReusableEventsBindingTest : public ::testing::Test {
protected:
  void SetUp() override {
    NextEventHandle = 0x1000;
    CreatedEvents.clear();
    RetainCounts.clear();
    ReleaseCounts.clear();
    KernelLaunchWaitLists.clear();
    KernelEvents.clear();
    BarrierWaitLists.clear();
    CommandBufferWaitLists.clear();
    CommandBufferEvents.clear();
    WaitedEvents.clear();

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
    mock::getCallbacks().set_replace_callback(
        "urEnqueueKernelLaunchWithArgsExp",
        &redefinedUrEnqueueKernelLaunchWithArgsExp);
    mock::getCallbacks().set_replace_callback("urEnqueueEventsWait",
                                              &redefinedUrEnqueueEventsWait);
    mock::getCallbacks().set_replace_callback(
        "urEnqueueCommandBufferExp", &redefinedUrEnqueueCommandBufferExp);
    mock::getCallbacks().set_replace_callback(
        "urEnqueueEventsWaitWithBarrierExt",
        &redefinedUrEnqueueEventsWaitWithBarrierExt);
    mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                            &after_urDeviceGetInfo);

    Dev = sycl::platform().get_devices()[0];
    Ctx = sycl::context{Dev};
  }

  sycl::queue inOrderQueue() {
    return sycl::queue{Ctx, Dev, sycl::property::queue::in_order{}};
  }

  sycl::unittest::UrMock<> Mock;
  sycl::device Dev;
  sycl::context Ctx;
};

// Nothing depends on the previous signal: the binding and its backend event
// are used again.
TEST_F(ReusableEventsBindingTest, SignalWithoutDependentsReusesBackendEvent) {
  sycl::queue Q = inOrderQueue();
  sycl::event E = syclex::make_event(Ctx);

  syclex::enqueue_signal_event(Q, E);
  ASSERT_EQ(CreatedEvents.size(), 1u);
  const ur_event_handle_t First = CreatedEvents[0];
  EXPECT_EQ(handleOf(E), First);

  syclex::enqueue_signal_event(Q, E);
  EXPECT_EQ(CreatedEvents.size(), 1u);
  EXPECT_EQ(handleOf(E), First);
  EXPECT_EQ(releases(First), 0);

  Q.wait();
}

// A kernel held behind a host task keeps waiting for the signal it was
// submitted with, although the event has been signaled again since. The
// second signal gets a backend event of its own.
TEST_F(ReusableEventsBindingTest, HeldKernelWaitsForCapturedSignal) {
  sycl::queue SignalQueue = inOrderQueue();
  sycl::queue Q = inOrderQueue();
  sycl::event E = syclex::make_event(Ctx);

  syclex::enqueue_signal_event(SignalQueue, E);
  const ur_event_handle_t First = CreatedEvents.back();

  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};
  blockQueue(Q, Gate);

  // Held in the runtime behind the host task; the dependency is captured now.
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    CGH.single_task<BindingTestKernel>([]() {});
  });

  syclex::enqueue_signal_event(SignalQueue, E);
  ASSERT_EQ(CreatedEvents.size(), 2u);
  const ur_event_handle_t Second = CreatedEvents.back();
  ASSERT_NE(First, Second);
  EXPECT_EQ(handleOf(E), Second);
  // The captured dependency keeps the first signal alive.
  EXPECT_EQ(releases(First), 0);

  Gate->open();
  Q.wait();
  SignalQueue.wait();

  ASSERT_EQ(KernelLaunchWaitLists.size(), 1u);
  EXPECT_EQ(KernelLaunchWaitLists[0], std::vector<ur_event_handle_t>{First});
  EXPECT_LE(releases(First), 1);
}

// The same for a barrier with a wait list, which keeps that list outside of the
// command group dependencies and resolves it when it is enqueued.
TEST_F(ReusableEventsBindingTest, HeldBarrierWaitsForCapturedSignal) {
  sycl::queue SignalQueue = inOrderQueue();
  sycl::queue Q = inOrderQueue();
  sycl::event E = syclex::make_event(Ctx);

  syclex::enqueue_signal_event(SignalQueue, E);
  const ur_event_handle_t First = CreatedEvents.back();

  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};
  blockQueue(Q, Gate);

  Q.submit([&](sycl::handler &CGH) { CGH.ext_oneapi_barrier({E}); });

  syclex::enqueue_signal_event(SignalQueue, E);
  ASSERT_EQ(CreatedEvents.size(), 2u);
  ASSERT_NE(CreatedEvents.back(), First);

  Gate->open();
  Q.wait();
  SignalQueue.wait();

  const auto Barriers = barriersWithWaitList();
  ASSERT_EQ(Barriers.size(), 1u);
  EXPECT_EQ(Barriers[0], std::vector<ur_event_handle_t>{First});
}

// A host task reads its dependencies when it runs, on the thread pool. It too
// waits for the signal it was submitted with.
TEST_F(ReusableEventsBindingTest, HostTaskWaitsForCapturedSignal) {
  sycl::queue SignalQueue = inOrderQueue();
  sycl::queue Q = inOrderQueue();
  sycl::event E = syclex::make_event(Ctx);

  syclex::enqueue_signal_event(SignalQueue, E);
  const ur_event_handle_t First = CreatedEvents.back();

  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};
  blockQueue(Q, Gate);

  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    CGH.host_task([] {});
  });

  syclex::enqueue_signal_event(SignalQueue, E);
  ASSERT_EQ(CreatedEvents.size(), 2u);
  const ur_event_handle_t Second = CreatedEvents.back();

  Gate->open();
  Q.wait();
  SignalQueue.wait();

  std::lock_guard<std::mutex> Lock(BackendMutex);
  EXPECT_NE(std::find(WaitedEvents.begin(), WaitedEvents.end(), First),
            WaitedEvents.end());
  EXPECT_EQ(std::find(WaitedEvents.begin(), WaitedEvents.end(), Second),
            WaitedEvents.end());
}

// An event which has never been enqueued for signaling is complete. A command
// which depends on it stays free of the dependency even if the event is
// enqueued for signaling before the command reaches the backend.
TEST_F(ReusableEventsBindingTest, UnsignaledDependencyStaysComplete) {
  sycl::queue SignalQueue = inOrderQueue();
  sycl::queue Q = inOrderQueue();
  sycl::event E = syclex::make_event(Ctx);

  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};
  blockQueue(Q, Gate);

  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    CGH.single_task<BindingTestKernel>([]() {});
  });

  syclex::enqueue_signal_event(SignalQueue, E);
  ASSERT_EQ(CreatedEvents.size(), 1u);
  const ur_event_handle_t Signal = CreatedEvents[0];
  EXPECT_EQ(handleOf(E), Signal);

  Gate->open();
  Q.wait();
  SignalQueue.wait();

  // Neither the backend nor the host waited for the signal.
  ASSERT_EQ(KernelLaunchWaitLists.size(), 1u);
  EXPECT_TRUE(KernelLaunchWaitLists[0].empty());
  std::lock_guard<std::mutex> Lock(BackendMutex);
  EXPECT_EQ(std::find(WaitedEvents.begin(), WaitedEvents.end(), Signal),
            WaitedEvents.end());
}

// An event returned by a submission whose command is still held in the runtime
// is enqueued for signaling elsewhere. The command keeps its own binding, and
// so does the queue's in-order bookkeeping: the next submission to the queue
// depends on the kernel, not on the signal, and queue::wait covers the kernel.
TEST_F(ReusableEventsBindingTest, PendingProducerKeepsItsBinding) {
  sycl::queue Q = inOrderQueue();
  sycl::queue SignalQueue = inOrderQueue();

  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};
  blockQueue(Q, Gate);
  sycl::event E = Q.submit(
      [&](sycl::handler &CGH) { CGH.single_task<BindingTestKernel>([]() {}); });
  EXPECT_EQ(handleOf(E), nullptr);

  syclex::enqueue_signal_event(SignalQueue, E);
  ASSERT_EQ(CreatedEvents.size(), 1u);
  const ur_event_handle_t Signal = CreatedEvents[0];
  EXPECT_EQ(handleOf(E), Signal);

  // The next command on the queue depends on the kernel, which is still held,
  // so it is held too. Depending on the signal instead would let it through.
  Q.submit(
      [&](sycl::handler &CGH) { CGH.single_task<BindingTestKernel>([]() {}); });
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    EXPECT_TRUE(KernelLaunchWaitLists.empty());
  }

  Gate->open();
  Q.wait();
  SignalQueue.wait();

  // Both kernels reached the backend, in order, and the kernel's backend event
  // did not replace the signal's.
  ASSERT_EQ(KernelEvents.size(), 2u);
  EXPECT_NE(KernelEvents[0], Signal);
  EXPECT_EQ(handleOf(E), Signal);
  EXPECT_EQ(releases(Signal), 0);
}

// The same for an out-of-order queue: a barrier submitted after the re-signal
// waits for the held kernel, which the queue remembers as the signal it was.
TEST_F(ReusableEventsBindingTest, BarrierAfterResignalWaitsForPendingKernel) {
  sycl::queue Q{Ctx, Dev};
  sycl::queue SignalQueue = inOrderQueue();

  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};
  sycl::event HostTask = blockQueue(Q, Gate);
  sycl::event E = Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(HostTask);
    CGH.single_task<BindingTestKernel>([]() {});
  });
  EXPECT_EQ(handleOf(E), nullptr);

  syclex::enqueue_signal_event(SignalQueue, E);
  ASSERT_EQ(CreatedEvents.size(), 1u);
  const ur_event_handle_t Signal = CreatedEvents[0];

  Q.submit([&](sycl::handler &CGH) { CGH.ext_oneapi_barrier(); });

  Gate->open();
  Q.wait();
  SignalQueue.wait();

  ASSERT_EQ(KernelEvents.size(), 1u);
  const ur_event_handle_t Kernel = KernelEvents[0];
  EXPECT_NE(Kernel, Signal);
  const auto Barriers = barriersWithWaitList();
  ASSERT_EQ(Barriers.size(), 1u);
  EXPECT_EQ(Barriers[0], std::vector<ur_event_handle_t>{Kernel});
}

// event::get_wait_list reports the dependencies of the latest signal. A
// consumer which got its dependency straight into the backend does not keep
// the previous signal alive, so signaling again reuses the backend event.
TEST_F(ReusableEventsBindingTest, GetWaitListFollowsTheSignal) {
  sycl::queue Q{Ctx, Dev}; // out-of-order: bypass submissions record their deps
  sycl::event E = syclex::make_event(Ctx);

  syclex::enqueue_signal_event(Q, E);
  ASSERT_EQ(CreatedEvents.size(), 1u);
  const ur_event_handle_t First = CreatedEvents[0];

  sycl::event K = Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    CGH.single_task<BindingTestKernel>([]() {});
  });
  ASSERT_EQ(K.get_wait_list().size(), 1u);
  EXPECT_EQ(K.get_wait_list()[0], E);

  syclex::enqueue_signal_event(Q, E);
  EXPECT_EQ(CreatedEvents.size(), 1u);
  EXPECT_EQ(handleOf(E), First);
  EXPECT_TRUE(E.get_wait_list().empty());

  Q.wait();
}

// An updatable executable graph remembers its previous executions which went
// through the scheduler, so that the next execution runs after them. That
// memory is the execution's signal as it was: re-signaling the event the
// execution returned does not let the next execution overtake it.
TEST_F(ReusableEventsBindingTest,
       GraphExecutionAfterResignalWaitsForPreviousExecution) {
  sycl::queue Q{Ctx, Dev}; // out-of-order: no implicit in-order dependency
  sycl::queue SignalQueue = inOrderQueue();

  syclex::command_graph<syclex::graph_state::modifiable> Graph{Ctx, Dev};
  Graph.add(
      [&](sycl::handler &CGH) { CGH.single_task<BindingTestKernel>([]() {}); });
  auto Exec = Graph.finalize(syclex::property::graph::updatable{});

  // The first execution is held behind a host task, so it goes through the
  // scheduler and is remembered by the executable graph.
  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};
  sycl::event HostTask = blockQueue(Q, Gate);
  sycl::event E1 = Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(HostTask);
    CGH.ext_oneapi_graph(Exec);
  });
  EXPECT_EQ(handleOf(E1), nullptr);

  syclex::enqueue_signal_event(SignalQueue, E1);
  ASSERT_EQ(CreatedEvents.size(), 1u);
  const ur_event_handle_t Signal = CreatedEvents[0];
  EXPECT_EQ(handleOf(E1), Signal);

  // The second execution depends on the first one as recorded, which is still
  // held, so it is held too. Depending on the signal would let it through.
  Q.ext_oneapi_graph(Exec);
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    EXPECT_TRUE(CommandBufferWaitLists.empty());
  }

  Gate->open();
  Q.wait();
  SignalQueue.wait();

  std::lock_guard<std::mutex> Lock(BackendMutex);
  ASSERT_EQ(CommandBufferWaitLists.size(), 2u);
  ASSERT_EQ(CommandBufferEvents.size(), 2u);
  EXPECT_TRUE(CommandBufferWaitLists[0].empty());
  EXPECT_EQ(CommandBufferWaitLists[1],
            std::vector<ur_event_handle_t>{CommandBufferEvents[0]});
  EXPECT_EQ(handleOf(E1), Signal);
}

} // anonymous namespace
