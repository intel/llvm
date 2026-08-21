//==--- CrossContextProxyEvents.cpp --- Cross-context proxy event tests ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A dependency that crosses a context boundary cannot be expressed in the
// backend. When the consuming context supports host-signalled events the
// runtime creates an unsignalled proxy event in the consuming command's
// context, hands it to the adapter as an ordinary dependency and lets the host
// task thread pool signal it once the producing event has retired. Otherwise it
// falls back to connecting the two contexts with an empty host task, which
// holds the consuming command back in the graph.
//
// These tests drive that logic through the SYCL API and observe it at the UR
// boundary: which entry points are called, in which context a proxy is created,
// what ends up in the consuming command's wait list, and that every proxy is
// eventually signalled and released.
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>

#include <detail/context_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/global_handler.hpp>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace detail = sycl::detail;

class ProducerKernel;
class ConsumerKernel;
MOCK_INTEGRATION_HEADER(ProducerKernel)
MOCK_INTEGRATION_HEADER(ConsumerKernel)

sycl::unittest::MockDeviceImage Img =
    sycl::unittest::generateDefaultImage({"ProducerKernel", "ConsumerKernel"});
sycl::unittest::MockDeviceImageArray<1> ImgArray{&Img};

namespace {

// The proxy is signalled from a thread of the host task pool, so everything the
// mocks record has to be synchronized with the test body.
std::mutex Mutex;
std::condition_variable CV;

// All of the following is guarded by Mutex.
std::vector<ur_context_handle_t> ProxyCreateContexts;
std::vector<ur_event_handle_t> CreatedProxies;
std::vector<ur_event_handle_t> SignalledProxies;
std::vector<ur_event_handle_t> ReleasedEvents;
std::vector<std::vector<ur_event_handle_t>> LaunchWaitLists;
std::vector<std::string> AsyncExceptions;

// An async_handler for a queue whose commands are expected to fail. The default
// one terminates the process.
void recordAsyncExceptions(sycl::exception_list Exceptions) {
  std::lock_guard<std::mutex> Lock(Mutex);
  for (const std::exception_ptr &E : Exceptions) {
    try {
      std::rethrow_exception(E);
    } catch (const std::exception &Caught) {
      AsyncExceptions.emplace_back(Caught.what());
    }
  }
  CV.notify_all();
}

// A handle is a proxy if urEventCreateHostSignalExp is what handed it out. Call
// with Mutex held.
bool isProxyHandle(ur_event_handle_t Handle) {
  return std::find(CreatedProxies.begin(), CreatedProxies.end(), Handle) !=
         CreatedProxies.end();
}

// Knobs, set by a test before it submits anything and only read afterwards.
std::atomic<bool> HostSignalSupported{true};
std::atomic<ur_result_t> CreateProxyResult{UR_RESULT_SUCCESS};

// Waits until Pred holds, which is evaluated with Mutex held. Returns false on
// timeout, so that a test fails rather than hangs.
template <typename PredT> bool waitFor(PredT Pred) {
  std::unique_lock<std::mutex> Lock(Mutex);
  return CV.wait_for(Lock, std::chrono::seconds(20), Pred);
}

//===----------------------------------------------------------------------===//
// UR mocks
//===----------------------------------------------------------------------===//

// Every event handle in these tests is a reference-counted mock dummy handle,
// created by the mock adapter's own default implementations. That matters: a
// pool thread can outlive a test and hence the callbacks installed below, so
// its urEventRelease call may well land in the mock adapter's default
// implementation, which only handles dummy handles.
//
// Nothing here therefore replaces an entry point that hands out or frees a
// handle - the mocks only observe, and inject the one failure a test needs.

ur_result_t before_urEventCreateHostSignalExp(void *) {
  // A non-success return from a before-callback is the entry point's result,
  // the default implementation is skipped.
  return CreateProxyResult.load();
}

ur_result_t after_urEventCreateHostSignalExp(void *pParams) {
  auto params =
      *static_cast<ur_event_create_host_signal_exp_params_t *>(pParams);

  std::lock_guard<std::mutex> Lock(Mutex);
  CreatedProxies.push_back(**params.pphEvent);
  ProxyCreateContexts.push_back(*params.phContext);
  CV.notify_all();
  return UR_RESULT_SUCCESS;
}

ur_result_t after_urEventHostSignalExp(void *pParams) {
  auto params = *static_cast<ur_event_host_signal_exp_params_t *>(pParams);

  std::lock_guard<std::mutex> Lock(Mutex);
  SignalledProxies.push_back(*params.phEvent);
  CV.notify_all();
  return UR_RESULT_SUCCESS;
}

ur_result_t after_urEnqueueKernelLaunchWithArgsExp(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_kernel_launch_with_args_exp_params_t *>(pParams);

  std::vector<ur_event_handle_t> WaitList;
  const uint32_t NumEvents = *params.pnumEventsInWaitList;
  if (const ur_event_handle_t *Events = *params.pphEventWaitList)
    WaitList.assign(Events, Events + NumEvents);

  std::lock_guard<std::mutex> Lock(Mutex);
  LaunchWaitLists.push_back(std::move(WaitList));
  CV.notify_all();
  return UR_RESULT_SUCCESS;
}

// Runs after the default implementation has dropped a reference, so the handle
// may already be gone - it is only recorded, never dereferenced.
ur_result_t after_urEventRelease(void *pParams) {
  auto params = *static_cast<ur_event_release_params_t *>(pParams);

  std::lock_guard<std::mutex> Lock(Mutex);
  ReleasedEvents.push_back(*params.phEvent);
  CV.notify_all();
  return UR_RESULT_SUCCESS;
}

// A pool job waits for the dependency with urEventWait, which the mock adapter
// completes immediately. These two knobs, both guarded by Mutex, are what lets
// a test control that wait: waiting for GatedEvent blocks until the gate is
// cleared, and waiting for FailedEvent fails.
ur_event_handle_t GatedEvent = nullptr;
ur_event_handle_t FailedEvent = nullptr;

// Call with Mutex held.
bool inWaitList(const ur_event_handle_t *Events, uint32_t NumEvents,
                ur_event_handle_t Handle) {
  return Handle &&
         std::find(Events, Events + NumEvents, Handle) != Events + NumEvents;
}

ur_result_t before_urEventWait(void *pParams) {
  auto params = *static_cast<ur_event_wait_params_t *>(pParams);
  const uint32_t NumEvents = *params.pnumEvents;
  const ur_event_handle_t *Events = *params.pphEventWaitList;

  std::unique_lock<std::mutex> Lock(Mutex);
  // Bounded, so that a test that forgets to open the gate fails rather than
  // hangs.
  CV.wait_for(Lock, std::chrono::seconds(20),
              [&] { return !inWaitList(Events, NumEvents, GatedEvent); });

  if (inWaitList(Events, NumEvents, FailedEvent))
    return UR_RESULT_ERROR_UNKNOWN;
  return UR_RESULT_SUCCESS;
}

// Sets the handle whose wait blocks; pass nullptr to let the waiters through.
void gateWaitFor(ur_event_handle_t Handle) {
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    GatedEvent = Handle;
  }
  CV.notify_all();
}

ur_result_t after_urDeviceGetInfo(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);

  if (*params.ppropName != UR_DEVICE_INFO_HOST_SIGNAL_EVENT_SUPPORT_EXP)
    return UR_RESULT_SUCCESS;

  if (*params.ppPropValue)
    *static_cast<ur_bool_t *>(*params.ppPropValue) =
        ur_bool_t{HostSignalSupported.load()};
  if (*params.ppPropSizeRet)
    **params.ppPropSizeRet = sizeof(ur_bool_t);
  return UR_RESULT_SUCCESS;
}

//===----------------------------------------------------------------------===//
// Test fixture
//===----------------------------------------------------------------------===//

class CrossContextProxyEventsTest : public ::testing::Test {
protected:
  void SetUp() override {
    {
      std::lock_guard<std::mutex> Lock(Mutex);
      ProxyCreateContexts.clear();
      CreatedProxies.clear();
      SignalledProxies.clear();
      ReleasedEvents.clear();
      LaunchWaitLists.clear();
      AsyncExceptions.clear();
      GatedEvent = nullptr;
      FailedEvent = nullptr;
    }
    HostSignalSupported = true;
    CreateProxyResult = UR_RESULT_SUCCESS;

    auto &Callbacks = mock::getCallbacks();
    Callbacks.set_before_callback("urEventCreateHostSignalExp",
                                  &before_urEventCreateHostSignalExp);
    Callbacks.set_after_callback("urEventCreateHostSignalExp",
                                 &after_urEventCreateHostSignalExp);
    Callbacks.set_after_callback("urEventHostSignalExp",
                                 &after_urEventHostSignalExp);
    Callbacks.set_after_callback("urEnqueueKernelLaunchWithArgsExp",
                                 &after_urEnqueueKernelLaunchWithArgsExp);
    Callbacks.set_after_callback("urEventRelease", &after_urEventRelease);
    Callbacks.set_before_callback("urEventWait", &before_urEventWait);
    Callbacks.set_after_callback("urDeviceGetInfo", &after_urDeviceGetInfo);
  }

  void TearDown() override {
    // Let go of whatever a pool job is still waiting for, then wait for the
    // pool to run dry. The pool outlives this test, so without this the events
    // its jobs hold - and the UR calls they make on them - would spill into the
    // next test, after the mock callbacks below have been reset.
    gateWaitFor(nullptr);
    detail::GlobalHandler::instance().drainThreadPool();

    // An abandoned proxy would keep a device submission channel blocked in a
    // real run. Whatever a test does, every proxy it created has to end up
    // signalled.
    std::lock_guard<std::mutex> Lock(Mutex);
    EXPECT_EQ(SignalledProxies.size(), CreatedProxies.size())
        << "a cross-context proxy event was left unsignalled";
  }

  sycl::unittest::UrMock<> Mock;
};

// Two explicitly created contexts on the same device, so that a dependency
// between their queues crosses a context boundary. The mock adapter hands out a
// distinct handle per urContextCreate.
struct TwoContexts {
  sycl::device Dev;
  sycl::context Ctx1;
  sycl::context Ctx2;
  sycl::queue Q1;
  sycl::queue Q2;

  TwoContexts()
      : Dev(sycl::platform().get_devices()[0]), Ctx1(Dev), Ctx2(Dev),
        Q1(Ctx1, Dev), Q2(Ctx2, Dev) {}
};

ur_context_handle_t getHandle(const sycl::context &Ctx) {
  return detail::getSyclObjImpl(Ctx)->getHandleRef();
}

sycl::event submitProducer(sycl::queue &Q) {
  return Q.submit(
      [&](sycl::handler &CGH) { CGH.single_task<ProducerKernel>([]() {}); });
}

sycl::event submitConsumer(sycl::queue &Q,
                           const std::vector<sycl::event> &Deps) {
  return Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Deps);
    CGH.single_task<ConsumerKernel>([]() {});
  });
}

// Runs the producer to completion and forgets its launch, so that the next
// recorded wait list is unambiguously the consumer's.
sycl::event runProducer(sycl::queue &Q) {
  sycl::event E = submitProducer(Q);
  Q.wait();

  std::lock_guard<std::mutex> Lock(Mutex);
  LaunchWaitLists.clear();
  return E;
}

//===----------------------------------------------------------------------===//
// The proxy path
//===----------------------------------------------------------------------===//

// A cross-context dependency creates exactly one proxy, and it is created in
// the *consuming* command's context - that is the whole point, the adapter can
// only wait on an event of its own context.
TEST_F(CrossContextProxyEventsTest, ProxyIsCreatedInConsumerContext) {
  TwoContexts T;
  ASSERT_NE(T.Ctx1, T.Ctx2);

  sycl::event E1 = runProducer(T.Q1);
  submitConsumer(T.Q2, {E1});
  T.Q2.wait();

  std::lock_guard<std::mutex> Lock(Mutex);
  ASSERT_EQ(CreatedProxies.size(), 1u);
  ASSERT_EQ(ProxyCreateContexts.size(), 1u);
  EXPECT_EQ(ProxyCreateContexts[0], getHandle(T.Ctx2));
  EXPECT_NE(ProxyCreateContexts[0], getHandle(T.Ctx1));
}

// The consuming command is handed to the adapter waiting on the proxy, and
// *only* on the proxy - the producing event belongs to a foreign context and
// must not appear in the wait list.
TEST_F(CrossContextProxyEventsTest, ConsumerWaitsOnProxyOnly) {
  TwoContexts T;

  sycl::event E1 = runProducer(T.Q1);
  const ur_event_handle_t ProducerHandle =
      detail::getSyclObjImpl(E1)->getHandle();

  submitConsumer(T.Q2, {E1});
  T.Q2.wait();

  std::lock_guard<std::mutex> Lock(Mutex);
  ASSERT_EQ(CreatedProxies.size(), 1u);
  ASSERT_EQ(LaunchWaitLists.size(), 1u);

  const std::vector<ur_event_handle_t> &WaitList = LaunchWaitLists[0];
  ASSERT_EQ(WaitList.size(), 1u);
  EXPECT_EQ(WaitList[0], CreatedProxies[0]);
  EXPECT_TRUE(isProxyHandle(WaitList[0]));
  EXPECT_NE(WaitList[0], ProducerHandle);
}

// The proxy is unsignalled when it is created, and a pool thread signals it
// once the producing event has retired.
TEST_F(CrossContextProxyEventsTest, ProxyIsSignalledOnceDependencyRetires) {
  TwoContexts T;

  sycl::event E1 = runProducer(T.Q1);
  submitConsumer(T.Q2, {E1});
  T.Q2.wait();

  ASSERT_TRUE(waitFor([] { return !SignalledProxies.empty(); }));

  std::lock_guard<std::mutex> Lock(Mutex);
  ASSERT_EQ(CreatedProxies.size(), 1u);
  ASSERT_EQ(SignalledProxies.size(), 1u);
  EXPECT_EQ(SignalledProxies[0], CreatedProxies[0]);
}

// Each cross-context dependency gets its own proxy, all of them created in the
// consuming context.
TEST_F(CrossContextProxyEventsTest, EachForeignDependencyGetsItsOwnProxy) {
  sycl::device Dev = sycl::platform().get_devices()[0];
  sycl::context Ctx1{Dev}, Ctx2{Dev}, Ctx3{Dev};
  sycl::queue Q1{Ctx1, Dev}, Q2{Ctx2, Dev}, Q3{Ctx3, Dev};

  sycl::event E1 = submitProducer(Q1);
  sycl::event E2 = submitProducer(Q2);
  Q1.wait();
  Q2.wait();
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    LaunchWaitLists.clear();
  }

  submitConsumer(Q3, {E1, E2});
  Q3.wait();

  ASSERT_TRUE(waitFor([] { return SignalledProxies.size() == 2u; }));

  std::lock_guard<std::mutex> Lock(Mutex);
  ASSERT_EQ(CreatedProxies.size(), 2u);
  EXPECT_NE(CreatedProxies[0], CreatedProxies[1]);
  for (ur_context_handle_t Handle : ProxyCreateContexts)
    EXPECT_EQ(Handle, getHandle(Ctx3));

  ASSERT_EQ(LaunchWaitLists.size(), 1u);
  const std::vector<ur_event_handle_t> &WaitList = LaunchWaitLists[0];
  EXPECT_EQ(WaitList.size(), 2u);
  for (ur_event_handle_t Handle : WaitList)
    EXPECT_TRUE(isProxyHandle(Handle));
}

// The proxy is owned by the consuming command and released once that command no
// longer needs it. Draining the scheduler is what a shutdown would do.
TEST_F(CrossContextProxyEventsTest, ProxyIsReleased) {
  {
    TwoContexts T;

    sycl::event E1 = runProducer(T.Q1);
    submitConsumer(T.Q2, {E1});
    T.Q2.wait();

    ASSERT_TRUE(waitFor([] { return !SignalledProxies.empty(); }));
  }

  detail::GlobalHandler::instance().prepareSchedulerToRelease(true);

  EXPECT_TRUE(waitFor([] {
    return CreatedProxies.size() == 1u &&
           std::find(ReleasedEvents.begin(), ReleasedEvents.end(),
                     CreatedProxies[0]) != ReleasedEvents.end();
  })) << "the proxy event was never released";
}

//===----------------------------------------------------------------------===//
// Fallback to the host task connection
//===----------------------------------------------------------------------===//

// When the consuming context cannot host-signal an event the runtime must keep
// using the host task connection: the device query gates the whole mechanism.
TEST_F(CrossContextProxyEventsTest, UnsupportedDeviceFallsBackToHostTask) {
  HostSignalSupported = false;

  TwoContexts T;

  sycl::event E1 = runProducer(T.Q1);
  submitConsumer(T.Q2, {E1});
  T.Q1.wait();
  T.Q2.wait();

  std::lock_guard<std::mutex> Lock(Mutex);
  EXPECT_TRUE(CreatedProxies.empty());
  for (const std::vector<ur_event_handle_t> &WaitList : LaunchWaitLists)
    for (ur_event_handle_t Handle : WaitList)
      EXPECT_FALSE(isProxyHandle(Handle));
}

// The device may claim support and the adapter still refuse to create the
// event. That must degrade to the host task connection rather than fail the
// submission.
TEST_F(CrossContextProxyEventsTest, CreateFailureFallsBackToHostTask) {
  CreateProxyResult = UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  TwoContexts T;

  sycl::event E1 = runProducer(T.Q1);
  EXPECT_NO_THROW(submitConsumer(T.Q2, {E1}));
  T.Q1.wait();
  T.Q2.wait();

  std::lock_guard<std::mutex> Lock(Mutex);
  EXPECT_TRUE(CreatedProxies.empty());
  for (const std::vector<ur_event_handle_t> &WaitList : LaunchWaitLists)
    for (ur_event_handle_t Handle : WaitList)
      EXPECT_FALSE(isProxyHandle(Handle));
}

// A dependency inside one context needs no proxy - the adapter can wait on the
// producing event directly.
TEST_F(CrossContextProxyEventsTest, SameContextDependencyNeedsNoProxy) {
  sycl::device Dev = sycl::platform().get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Q1{Ctx, Dev}, Q2{Ctx, Dev};

  sycl::event E1 = submitProducer(Q1);
  Q1.wait();
  submitConsumer(Q2, {E1});
  Q2.wait();

  std::lock_guard<std::mutex> Lock(Mutex);
  EXPECT_TRUE(CreatedProxies.empty());
}

//===----------------------------------------------------------------------===//
// The "every proxy is eventually signalled" guarantee
//===----------------------------------------------------------------------===//

// A dependency that has not retired yet must not have its proxy signalled -
// that is what makes the proxy an actual dependency rather than a no-op.
TEST_F(CrossContextProxyEventsTest, ProxyStaysUnsignalledWhileDependencyRuns) {
  TwoContexts T;

  sycl::event E1 = submitProducer(T.Q1);
  // Whoever waits for the dependency now blocks, so the only way past this gate
  // is the one the mechanism is supposed to take.
  gateWaitFor(detail::getSyclObjImpl(E1)->getHandle());

  submitConsumer(T.Q2, {E1});
  ASSERT_TRUE(waitFor([] { return !CreatedProxies.empty(); }));

  {
    std::lock_guard<std::mutex> Lock(Mutex);
    EXPECT_TRUE(SignalledProxies.empty())
        << "a proxy must not be signalled before its dependency retires";
  }

  // Let the dependency retire, the pool then has to catch up.
  gateWaitFor(nullptr);
  EXPECT_TRUE(
      waitFor([] { return SignalledProxies.size() == CreatedProxies.size(); }));

  T.Q1.wait();
  T.Q2.wait();
}

// The consuming command is in the adapter's hands by the time the pool starts
// waiting, so an outstanding proxy keeps a device submission channel blocked.
// Draining the pool - which is what the runtime does before it releases the
// scheduler - is therefore what guarantees that every proxy gets signalled:
// once the drain returns, no wait is left pending.
TEST_F(CrossContextProxyEventsTest, DrainingThePoolSignalsOutstandingProxies) {
  TwoContexts T;

  sycl::event E1 = submitProducer(T.Q1);
  gateWaitFor(detail::getSyclObjImpl(E1)->getHandle());

  submitConsumer(T.Q2, {E1});
  ASSERT_TRUE(waitFor([] { return !CreatedProxies.empty(); }));
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    ASSERT_TRUE(SignalledProxies.empty());
  }

  gateWaitFor(nullptr);
  detail::GlobalHandler::instance().drainThreadPool();

  // No waiting for the pool here - the drain is the synchronization.
  std::lock_guard<std::mutex> Lock(Mutex);
  ASSERT_EQ(CreatedProxies.size(), 1u);
  ASSERT_EQ(SignalledProxies.size(), 1u);
  EXPECT_EQ(SignalledProxies[0], CreatedProxies[0]);
}

// A failing wait must not strand the consuming command: it is in the adapter's
// hands already, so the proxy is signalled either way and the failure goes to
// the async handler of the queue the command was submitted to - which is where
// a host task connection would have reported it.
TEST_F(CrossContextProxyEventsTest, ProxyIsSignalledEvenIfTheWaitFails) {
  sycl::device Dev = sycl::platform().get_devices()[0];
  sycl::context Ctx1{Dev}, Ctx2{Dev};
  sycl::queue Q1{Ctx1, Dev};
  // Without a handler of its own the exception would end up in the default one,
  // which terminates the process.
  sycl::queue Q2{Ctx2, Dev, &recordAsyncExceptions};

  sycl::event E1 = runProducer(Q1);
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    FailedEvent = detail::getSyclObjImpl(E1)->getHandle();
    ASSERT_NE(FailedEvent, nullptr);
  }

  submitConsumer(Q2, {E1});
  ASSERT_TRUE(waitFor([] { return !SignalledProxies.empty(); }));

  {
    std::lock_guard<std::mutex> Lock(Mutex);
    // Cleared while the lock is held, i.e. before anything else waits for the
    // producer again: only the wait above was meant to fail.
    FailedEvent = nullptr;
    ASSERT_EQ(CreatedProxies.size(), 1u);
    ASSERT_EQ(SignalledProxies.size(), 1u);
    EXPECT_EQ(SignalledProxies[0], CreatedProxies[0]);
  }

  // Hands the recorded exception to Q2's handler. Has to happen here, while
  // that handler is still alive.
  Q2.wait_and_throw();

  std::lock_guard<std::mutex> Lock(Mutex);
  EXPECT_FALSE(AsyncExceptions.empty())
      << "a failed wait for a cross-context dependency was swallowed";
}

} // namespace
