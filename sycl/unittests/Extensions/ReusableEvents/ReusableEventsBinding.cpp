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

#include "ReusableEventsFixture.hpp"

#include <atomic>
#include <functional>
#include <string>
#include <tuple>

namespace {

using namespace reusable_events_test;

using ReusableEventsBindingTest = ReusableEventsTest;
using ReusableEventsBindingSupportTest = ReusableEventsSupportTest;

INSTANTIATE_TEST_SUITE_P(ReusableEventsBinding,
                         ReusableEventsBindingSupportTest, ::testing::Bool(),
                         supportName);

// Signals E on Q; the backend event of the signal stays pending until the test
// completes it.
void signalPending(sycl::queue &Q, sycl::event &E) {
  setBarriersStayPending(true);
  syclex::enqueue_signal_event(Q, E);
  setBarriersStayPending(false);
}

// A queue is empty when none of the pending backend events was enqueued to
// it. The mock otherwise leaves UR_QUEUE_INFO_EMPTY unanswered.
ur_result_t after_urQueueGetInfoEmpty(void *pParams) {
  auto params = *static_cast<ur_queue_get_info_params_t *>(pParams);
  if (*params.ppropName != UR_QUEUE_INFO_EMPTY || !*params.ppPropValue)
    return UR_RESULT_SUCCESS;
  const ur_queue_handle_t Queue = *params.phQueue;
  std::lock_guard<std::mutex> Lock(BackendMutex);
  *static_cast<ur_bool_t *>(*params.ppPropValue) = std::none_of(
      PendingEvents.begin(), PendingEvents.end(), [&](ur_event_handle_t E) {
        auto It = HandleQueues.find(E);
        return It != HandleQueues.end() && It->second == Queue;
      });
  return UR_RESULT_SUCCESS;
}

bool contains(const std::vector<sycl::event> &Events, const sycl::event &E) {
  return std::find(Events.begin(), Events.end(), E) != Events.end();
}

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
// tests-10-02 U03, the signal through the scheduler bypass.
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

// tests-10-02 U03. The same, with the first signal held behind a host task of
// its own: the consumer is released first, and must not wait for the signal,
// which is still held in the runtime. MakeEvent creates the unsignaled event
// in the given context. Returns the backend event of the signal.
ur_event_handle_t
heldConsumerOfFirstSignal(const sycl::context &C,
                          std::function<sycl::event()> MakeEvent) {
  const sycl::device Dev = C.get_devices()[0];
  const size_t FirstLaunch = kernelLaunches();
  ur_event_handle_t Signal = nullptr;
  {
    sycl::queue Q{C, Dev, sycl::property::queue::in_order{}};
    sycl::queue SignalQueue{C, Dev, sycl::property::queue::in_order{}};
    sycl::event E = MakeEvent();
    EXPECT_TRUE(isComplete(E));

    std::shared_ptr<HostTaskGate> Gates[2] = {std::make_shared<HostTaskGate>(),
                                              std::make_shared<HostTaskGate>()};
    OpenAtScopeExit OpenGates[2] = {{Gates[0]}, {Gates[1]}};
    const std::shared_ptr<HostTaskGate> &ConsumerGate = Gates[0];
    const std::shared_ptr<HostTaskGate> &SignalGate = Gates[1];

    blockQueue(Q, ConsumerGate);
    Q.submit([&](sycl::handler &CGH) {
      CGH.depends_on(E);
      CGH.single_task<BindingTestKernel>([]() {});
    });

    blockQueue(SignalQueue, SignalGate);
    syclex::enqueue_signal_event(SignalQueue, E);
    std::shared_ptr<sycl::detail::event_binding> SignalBinding = bindingOf(E);
    EXPECT_EQ(handleOf(E), nullptr);
    EXPECT_FALSE(isComplete(E));

    // The consumer reaches the backend with nothing to wait for, while the
    // signal is still held.
    ConsumerGate->open();
    const bool Launched = eventually(
        [&] { return KernelLaunchWaitLists.size() == FirstLaunch + 1; });
    EXPECT_TRUE(Launched) << "the consumer waits for the held signal";
    // Otherwise the queue wait would wait for the held signal forever.
    if (!Launched)
      SignalGate->open();
    Q.wait();
    {
      std::lock_guard<std::mutex> Lock(BackendMutex);
      if (KernelLaunchWaitLists.size() == FirstLaunch + 1) {
        EXPECT_TRUE(KernelLaunchWaitLists[FirstLaunch].empty());
      }
    }
    EXPECT_EQ(SignalBinding->getHandle(), nullptr);
    EXPECT_FALSE(isComplete(E));

    SignalGate->open();
    SignalQueue.wait();
    Signal = SignalBinding->getHandle();
    EXPECT_NE(Signal, nullptr);
    EXPECT_EQ(handleOf(E), Signal);
    EXPECT_TRUE(isComplete(E));
  }
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    EXPECT_TRUE(NullEntryWaitLists.empty());
  }
  if (Signal) {
    EXPECT_TRUE(eventually([&] { return ownershipBalancedLocked(Signal); }));
  }
  return Signal;
}

// tests-10-02 U03: an event made by make_event.
TEST_P(ReusableEventsBindingSupportTest,
       UnsignaledMadeEventStaysCompleteForHeldConsumer) {
  heldConsumerOfFirstSignal(Ctx, [&] { return syclex::make_event(Ctx); });
}

// tests-10-02 U03: an event made by make_event with properties.
TEST_P(ReusableEventsBindingSupportTest,
       UnsignaledEventWithPropertiesStaysCompleteForHeldConsumer) {
  const ur_event_handle_t Signal = heldConsumerOfFirstSignal(Ctx, [&] {
    return syclex::make_event(Ctx, syclex::properties{syclex::event_mode{
                                       syclex::event_mode_enum::low_power}});
  });
  if (GetParam()) {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    ASSERT_EQ(CreatedEvents.size(), 1u);
    EXPECT_EQ(CreatedEvents[0], Signal);
    EXPECT_TRUE(CreatedEventLowPower[0]);
  }
}

// tests-10-02 U03: a default constructed event, which gets the default context
// when it is first signaled.
TEST_P(ReusableEventsBindingSupportTest,
       UnsignaledDefaultConstructedEventStaysCompleteForHeldConsumer) {
  const sycl::context DefaultContext = sycl::queue{Dev}.get_context();
  heldConsumerOfFirstSignal(DefaultContext, [] { return sycl::event{}; });
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

// Whether the producer is a kernel (with a backend event) or an empty command
// group (enqueued without one), and whether the producer is released before
// the new signal.
class ReusableEventsOldProducerTest
    : public ReusableEventsTest,
      public ::testing::WithParamInterface<std::tuple<bool, bool>> {
protected:
  bool kernelProducer() const { return std::get<0>(GetParam()); }
  bool olderFirst() const { return std::get<1>(GetParam()); }
};

// tests-10-02 U05. The producer of E, held behind a host task, keeps the
// binding it was submitted with while E is signaled again behind a host task
// of its own. Neither signal publishes into nor completes the other one,
// whichever is released first.
TEST_P(ReusableEventsOldProducerTest, OldProducerKeepsToItsBinding) {
  sycl::queue Q = inOrderQueue();
  sycl::queue SignalQueue = inOrderQueue();
  sycl::queue Consumers = inOrderQueue();
  std::atomic<bool> ConsumerRan{false};
  std::future<void> QueueWait;
  std::shared_ptr<HostTaskGate> Gates[2] = {std::make_shared<HostTaskGate>(),
                                            std::make_shared<HostTaskGate>()};
  OpenAtScopeExit OpenGates[2] = {{Gates[0]}, {Gates[1]}};
  const std::shared_ptr<HostTaskGate> &ProducerGate = Gates[0];
  const std::shared_ptr<HostTaskGate> &SignalGate = Gates[1];
  CompleteAllAtScopeExit CompleteAll;
  setKernelsStayPending(true);

  blockQueue(Q, ProducerGate);
  sycl::event E = kernelProducer() ? Q.submit([&](sycl::handler &CGH) {
    CGH.single_task<BindingTestKernel>([]() {});
  })
                                   : Q.submit([](sycl::handler &) {});
  std::shared_ptr<sycl::detail::event_binding> S1 = bindingOf(E);
  ASSERT_EQ(S1->getHandle(), nullptr);
  ASSERT_NE(S1->MCommand, nullptr);

  // Depends on the producer.
  Consumers.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    CGH.host_task([&ConsumerRan] { ConsumerRan = true; });
  });

  blockQueue(SignalQueue, SignalGate);
  syclex::enqueue_signal_event(SignalQueue, E);
  std::shared_ptr<sycl::detail::event_binding> S2 = bindingOf(E);
  ASSERT_NE(S2, S1);
  auto *const S2Command = S2->MCommand;
  ASSERT_NE(S2Command, nullptr);
  auto ExpectS2Unchanged = [&] {
    EXPECT_EQ(bindingOf(E), S2);
    EXPECT_EQ(S2->getHandle(), nullptr);
    EXPECT_EQ(S2->MCommand, S2Command);
    EXPECT_FALSE(S2->isCompleted());
    EXPECT_EQ(handleOf(E), nullptr);
    EXPECT_FALSE(isComplete(E));
  };

  if (olderFirst()) {
    // The producer is enqueued into its own binding.
    ProducerGate->open();
    ASSERT_TRUE(eventually([&] { return S1->MIsEnqueued.load(); }));
    const ur_event_handle_t S1Handle = S1->getHandle();
    if (kernelProducer()) {
      {
        std::lock_guard<std::mutex> Lock(BackendMutex);
        ASSERT_EQ(KernelEvents.size(), 1u);
        EXPECT_EQ(S1Handle, KernelEvents[0]);
      }
      ExpectS2Unchanged();
      // The consumer waits for the producer's backend event.
      EXPECT_TRUE(eventually([&] {
        return std::find(WaitedEvents.begin(), WaitedEvents.end(), S1Handle) !=
               WaitedEvents.end();
      }));
      EXPECT_FALSE(ConsumerRan);
      EXPECT_FALSE(S1->isCompleted());
      complete(S1Handle);
    } else {
      EXPECT_EQ(S1Handle, nullptr);
    }

    // The producer completes its own binding, and lets its consumer run; E
    // stays incomplete for the new signal.
    EXPECT_TRUE(eventually([&] { return ConsumerRan.load(); }));
    EXPECT_TRUE(S1->isCompleted());
    ExpectS2Unchanged();

    SignalGate->open();
    SignalQueue.wait();
    EXPECT_NE(handleOf(E), nullptr);
    EXPECT_NE(handleOf(E), S1Handle);
    EXPECT_TRUE(isComplete(E));
    EXPECT_EQ(S1->getHandle(), S1Handle);
  } else {
    QueueWait = std::async(std::launch::async, [&Q] { Q.wait(); });

    // The new signal completes first. The original queue and the consumer
    // keep waiting for the producer.
    SignalGate->open();
    SignalQueue.wait();
    EXPECT_NE(handleOf(E), nullptr);
    EXPECT_TRUE(isComplete(E));
    EXPECT_EQ(S1->getHandle(), nullptr);
    EXPECT_FALSE(S1->isCompleted());
    // Best effort: the queue wait cannot be observed entering the wait.
    EXPECT_TRUE(stillBlocked(QueueWait));
    EXPECT_FALSE(ConsumerRan);

    ProducerGate->open();
    ASSERT_TRUE(eventually([&] { return S1->MIsEnqueued.load(); }));
    const ur_event_handle_t S1Handle = S1->getHandle();
    if (kernelProducer()) {
      EXPECT_NE(S1Handle, nullptr);
      EXPECT_NE(S1Handle, handleOf(E));
      EXPECT_TRUE(stillBlocked(QueueWait));
      EXPECT_FALSE(ConsumerRan);
      complete(S1Handle);
    } else {
      EXPECT_EQ(S1Handle, nullptr);
    }
    EXPECT_TRUE(finishes(QueueWait));
    EXPECT_TRUE(eventually([&] { return ConsumerRan.load(); }));
    EXPECT_TRUE(S1->isCompleted());
    EXPECT_TRUE(isComplete(E));
  }
  Consumers.wait();
}

INSTANTIATE_TEST_SUITE_P(
    ReusableEventsBinding, ReusableEventsOldProducerTest,
    ::testing::Combine(::testing::Bool(), ::testing::Bool()),
    [](const ::testing::TestParamInfo<std::tuple<bool, bool>> &Info) {
      return std::string(std::get<0>(Info.param) ? "Kernel" : "EmptyGroup") +
             (std::get<1>(Info.param) ? "OlderFirst" : "NewerFirst");
    });

// tests-10-02 U05: a host task completes the binding it was submitted with.
// The head does not specify signaling the event a host task returned, so the
// event is moved on to a new binding the way a signal does it, without one.
TEST_F(ReusableEventsBindingTest, HostTaskCompletesItsOwnBinding) {
  sycl::queue Q = inOrderQueue();
  sycl::queue SignalQueue = inOrderQueue();
  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};

  sycl::event HostTask = blockQueue(Q, Gate);
  std::shared_ptr<sycl::detail::event_binding> S1 = bindingOf(HostTask);
  ASSERT_NE(S1->MCommand, nullptr);
  // A host task event has no context, which a signal needs.
  sycl::detail::event_impl &HostTaskImpl =
      *sycl::detail::getSyclObjImpl(HostTask);
  HostTaskImpl.setContextImpl(
      sycl::detail::getSyclObjImpl(SignalQueue)->getContextImpl());
  HostTaskImpl.prepareForSignal(*sycl::detail::getSyclObjImpl(SignalQueue),
                                /*Deferred=*/false);
  // As the submission of the signal would.
  HostTaskImpl.setSubmittedQueue(&*sycl::detail::getSyclObjImpl(SignalQueue));
  std::shared_ptr<sycl::detail::event_binding> S2 = bindingOf(HostTask);
  ASSERT_NE(S2, S1);
  EXPECT_FALSE(S2->isCompleted());

  Gate->open();
  Q.wait();
  EXPECT_TRUE(S1->isCompleted());
  EXPECT_EQ(S1->getHandle(), nullptr);
  EXPECT_FALSE(S2->isCompleted());
  EXPECT_EQ(S2->getHandle(), nullptr);
  EXPECT_EQ(S2->MCommand, nullptr);
  // The event's status is not checked: without the signal's barrier, S2 has
  // no backend event, so the status reads as complete.
}

// How an in-order queue is used after the event its pending kernel returned
// has been signaled again elsewhere.
enum class InOrderFollower { ReturnedEvent, Eventless, HostTaskThenEventless };

class ReusableEventsInOrderFollowerTest
    : public ReusableEventsTest,
      public ::testing::WithParamInterface<InOrderFollower> {};

// tests-10-02 U10. The next submissions to the queue depend on the kernel,
// through the queue's in-order bookkeeping, and not on the signal, which
// completes first. The queue wait waits for the kernel too.
TEST_P(ReusableEventsInOrderFollowerTest, FollowersKeepTheOriginalLastSignal) {
  sycl::queue Q1 = inOrderQueue();
  sycl::queue Q2 = inOrderQueue();
  std::atomic<bool> HostRan{false};
  std::future<void> QueueWait;
  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};
  CompleteAllAtScopeExit CompleteAll;
  setKernelsStayPending(true);

  blockQueue(Q1, Gate);
  sycl::event E = Q1.submit(
      [&](sycl::handler &CGH) { CGH.single_task<BindingTestKernel>([]() {}); });
  std::shared_ptr<sycl::detail::event_binding> S1 = bindingOf(E);

  signalPending(Q2, E);
  const ur_event_handle_t Signal = handleOf(E);
  ASSERT_NE(bindingOf(E), S1);
  ASSERT_NE(Signal, nullptr);
  complete(Signal);
  EXPECT_TRUE(isComplete(E));

  const bool HostTask = GetParam() == InOrderFollower::HostTaskThenEventless;
  switch (GetParam()) {
  case InOrderFollower::ReturnedEvent:
    Q1.submit([&](sycl::handler &CGH) {
      CGH.single_task<BindingTestKernel>([]() {});
    });
    break;
  case InOrderFollower::Eventless:
    syclex::single_task<BindingTestKernel>(Q1, []() {});
    break;
  case InOrderFollower::HostTaskThenEventless:
    Q1.submit([&](sycl::handler &CGH) {
      CGH.host_task([&HostRan] { HostRan = true; });
    });
    syclex::single_task<BindingTestKernel>(Q1, []() {});
    break;
  }

  // The signal is complete; everything on Q1 still waits for the kernel.
  EXPECT_EQ(kernelLaunches(), 0u);
  QueueWait = std::async(std::launch::async, [&Q1] { Q1.wait(); });
  // Best effort: the queue wait cannot be observed entering the wait.
  EXPECT_TRUE(stillBlocked(QueueWait));
  EXPECT_FALSE(HostRan);

  // The kernel reaches the backend first, and stays pending.
  Gate->open();
  const size_t Launches = HostTask ? 1 : 2;
  ASSERT_TRUE(
      eventually([&] { return KernelLaunchWaitLists.size() == Launches; }));
  const ur_event_handle_t Kernel = S1->getHandle();
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    EXPECT_EQ(Kernel, KernelEvents[0]);
  }
  EXPECT_TRUE(stillBlocked(QueueWait));
  if (HostTask) {
    // The host task waits for the kernel on the host.
    EXPECT_TRUE(eventually([&] {
      return std::find(WaitedEvents.begin(), WaitedEvents.end(), Kernel) !=
             WaitedEvents.end();
    }));
    EXPECT_FALSE(HostRan);
    complete(Kernel);
    EXPECT_TRUE(eventually([&] { return HostRan.load(); }));
    ASSERT_TRUE(eventually([&] { return KernelLaunchWaitLists.size() == 2; }));
  }
  {
    // No follower waits for the signal.
    std::lock_guard<std::mutex> Lock(BackendMutex);
    for (const auto &WaitList : KernelLaunchWaitLists)
      EXPECT_EQ(std::find(WaitList.begin(), WaitList.end(), Signal),
                WaitList.end());
  }

  completeAll();
  EXPECT_TRUE(finishes(QueueWait));
  EXPECT_EQ(handleOf(E), Signal);
  EXPECT_TRUE(S1->isCompleted());
}

INSTANTIATE_TEST_SUITE_P(
    ReusableEventsBinding, ReusableEventsInOrderFollowerTest,
    ::testing::Values(InOrderFollower::ReturnedEvent,
                      InOrderFollower::Eventless,
                      InOrderFollower::HostTaskThenEventless),
    [](const ::testing::TestParamInfo<InOrderFollower> &Info) {
      switch (Info.param) {
      case InOrderFollower::ReturnedEvent:
        return std::string("ReturnedEvent");
      case InOrderFollower::Eventless:
        return std::string("Eventless");
      case InOrderFollower::HostTaskThenEventless:
        return std::string("HostTaskThenEventless");
      }
      return std::string();
    });

// tests-10-02 U12. An out-of-order queue holds several commands behind a host
// task, E's among them; E is signaled again elsewhere and completes, and the
// application lets go of all the events. The queue is not empty, and its wait
// waits for its own work, completed in an order of its own.
TEST_F(ReusableEventsBindingTest, OutOfOrderQueueWaitFindsDroppedWork) {
  mock::getCallbacks().set_after_callback("urQueueGetInfo",
                                          &after_urQueueGetInfoEmpty);
  std::weak_ptr<sycl::detail::event_binding> Bindings[3];
  {
    sycl::queue Q1{Ctx, Dev};
    sycl::queue Q2 = inOrderQueue();
    std::future<void> Waiter;
    std::shared_ptr<HostTaskGate> Gates[2] = {std::make_shared<HostTaskGate>(),
                                              std::make_shared<HostTaskGate>()};
    OpenAtScopeExit OpenGates[2] = {{Gates[0]}, {Gates[1]}};
    CompleteAllAtScopeExit CompleteAll;
    setKernelsStayPending(true);
    {
      sycl::event Held = blockQueue(Q1, Gates[0]);
      sycl::event K1 = Q1.submit([&](sycl::handler &CGH) {
        CGH.depends_on(Held);
        CGH.single_task<BindingTestKernel>([]() {});
      });
      sycl::event E = Q1.submit([&](sycl::handler &CGH) {
        CGH.depends_on(Held);
        CGH.single_task<BindingTestKernel>([]() {});
      });
      // A host-only binding, released by a gate of its own.
      sycl::event HostTask = Q1.submit([&](sycl::handler &CGH) {
        CGH.depends_on(Held);
        CGH.host_task([Gate = Gates[1]] { Gate->wait(); });
      });
      Bindings[0] = bindingOf(K1);
      Bindings[1] = bindingOf(E);
      Bindings[2] = bindingOf(HostTask);

      syclex::enqueue_signal_event(Q2, E);
      Q2.wait();
      EXPECT_TRUE(isComplete(E));
      EXPECT_FALSE(Q1.ext_oneapi_empty());
    }
    // The commands still own their bindings.
    for (const auto &Binding : Bindings)
      EXPECT_FALSE(Binding.expired());
    EXPECT_FALSE(Q1.ext_oneapi_empty());

    Waiter = std::async(std::launch::async, [&Q1] { Q1.wait(); });
    // Best effort: the waiter cannot be observed entering the wait.
    EXPECT_TRUE(stillBlocked(Waiter));

    Gates[0]->open();
    ASSERT_TRUE(eventually([&] { return KernelEvents.size() == 2; }));
    // The commands of the kernels are cleaned up once they are in the backend,
    // and nobody else owns their bindings, so the handles are taken from the
    // launches. Which kernel is launched first does not matter.
    ur_event_handle_t K1Handle = nullptr;
    ur_event_handle_t EHandle = nullptr;
    {
      std::lock_guard<std::mutex> Lock(BackendMutex);
      K1Handle = KernelEvents[0];
      EHandle = KernelEvents[1];
    }

    // One kernel completes first, then the host task.
    complete(EHandle);
    EXPECT_TRUE(Gates[1]->waitEntered());
    EXPECT_TRUE(stillBlocked(Waiter));
    EXPECT_FALSE(Q1.ext_oneapi_empty());
    Gates[1]->open();

    // The wait reaches the first kernel, which is still running.
    const ur_queue_handle_t Q1Handle = handleOf(Q1);
    EXPECT_TRUE(eventually([&] {
      return std::find(WaitedEvents.begin(), WaitedEvents.end(), K1Handle) !=
                 WaitedEvents.end() ||
             std::find(FinishedQueues.begin(), FinishedQueues.end(),
                       Q1Handle) != FinishedQueues.end();
    }));
    EXPECT_TRUE(stillBlocked(Waiter));
    EXPECT_FALSE(Q1.ext_oneapi_empty());

    complete(K1Handle);
    EXPECT_TRUE(finishes(Waiter));
    EXPECT_TRUE(Q1.ext_oneapi_empty());
    // Waiting again finds nothing to wait for.
    Q1.wait_and_throw();
    Q1.wait();
    EXPECT_TRUE(Q1.ext_oneapi_empty());
  }
  EXPECT_TRUE(eventually([&] {
    return std::all_of(std::begin(Bindings), std::end(Bindings),
                       [](const auto &B) { return B.expired(); });
  }));
}

// tests-10-02 U12, the same for an in-order queue, whose wait and emptiness
// follow its last submission as captured: the host task, which waits for E's
// original kernel.
TEST_F(ReusableEventsBindingTest, InOrderQueueWaitFindsDroppedWork) {
  std::weak_ptr<sycl::detail::event_binding> Bindings[3];
  {
    sycl::queue Q1 = inOrderQueue();
    sycl::queue Q2 = inOrderQueue();
    std::future<void> Waiter;
    std::shared_ptr<HostTaskGate> Gates[2] = {std::make_shared<HostTaskGate>(),
                                              std::make_shared<HostTaskGate>()};
    OpenAtScopeExit OpenGates[2] = {{Gates[0]}, {Gates[1]}};
    CompleteAllAtScopeExit CompleteAll;
    setKernelsStayPending(true);
    {
      blockQueue(Q1, Gates[0]);
      sycl::event K1 = Q1.submit([&](sycl::handler &CGH) {
        CGH.single_task<BindingTestKernel>([]() {});
      });
      sycl::event E = Q1.submit([&](sycl::handler &CGH) {
        CGH.single_task<BindingTestKernel>([]() {});
      });
      sycl::event HostTask = Q1.submit([&](sycl::handler &CGH) {
        CGH.host_task([Gate = Gates[1]] { Gate->wait(); });
      });
      Bindings[0] = bindingOf(K1);
      Bindings[1] = bindingOf(E);
      Bindings[2] = bindingOf(HostTask);

      syclex::enqueue_signal_event(Q2, E);
      Q2.wait();
      EXPECT_TRUE(isComplete(E));
      EXPECT_FALSE(Q1.ext_oneapi_empty());
    }
    for (const auto &Binding : Bindings)
      EXPECT_FALSE(Binding.expired());
    EXPECT_FALSE(Q1.ext_oneapi_empty());

    Waiter = std::async(std::launch::async, [&Q1] { Q1.wait(); });
    // Best effort: the waiter cannot be observed entering the wait.
    EXPECT_TRUE(stillBlocked(Waiter));

    Gates[0]->open();
    ASSERT_TRUE(eventually([&] { return KernelLaunchWaitLists.size() == 2; }));
    ur_event_handle_t K1Handle = nullptr;
    ur_event_handle_t EHandle = nullptr;
    {
      std::lock_guard<std::mutex> Lock(BackendMutex);
      K1Handle = KernelEvents[0];
      EHandle = KernelEvents[1];
    }

    // The kernels complete in submission order; the host task waits for E's.
    complete(K1Handle);
    EXPECT_TRUE(eventually([&] {
      return std::find(WaitedEvents.begin(), WaitedEvents.end(), EHandle) !=
             WaitedEvents.end();
    }));
    EXPECT_TRUE(stillBlocked(Waiter));
    EXPECT_FALSE(Q1.ext_oneapi_empty());

    complete(EHandle);
    EXPECT_TRUE(Gates[1]->waitEntered());
    EXPECT_TRUE(stillBlocked(Waiter));
    EXPECT_FALSE(Q1.ext_oneapi_empty());

    Gates[1]->open();
    EXPECT_TRUE(finishes(Waiter));
    EXPECT_TRUE(Q1.ext_oneapi_empty());
    Q1.wait_and_throw();
    Q1.wait();
    EXPECT_TRUE(Q1.ext_oneapi_empty());
  }
  EXPECT_TRUE(eventually([&] {
    return std::all_of(std::begin(Bindings), std::end(Bindings),
                       [](const auto &B) { return B.expired(); });
  }));
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

// tests-10-02 U11. A signal on an out-of-order queue covers the queue's two
// held producers, a kernel and a USM copy, each behind a host task of its own.
// The kernels submitted after it, one returning an event and one through the
// eventless path, depend on that signal. E is signaled again elsewhere, and
// the new signal completes with only one producer released: the original
// signal and its followers keep waiting for the other one, and the followers
// then wait for the original signal, not for E's new one. A wait for the
// released producer alone does not wait for the other one.
TEST_F(ReusableEventsBindingTest, OutOfOrderSignalKeepsItsProducers) {
  sycl::queue Q{Ctx, Dev};
  sycl::queue Waits = inOrderQueue();
  sycl::queue SignalQueue = inOrderQueue();
  constexpr size_t Count = 16;
  int *Src = sycl::malloc_device<int>(Count, Q);
  int *Dst = sycl::malloc_device<int>(Count, Q);
  {
    std::shared_ptr<HostTaskGate> Gates[2] = {std::make_shared<HostTaskGate>(),
                                              std::make_shared<HostTaskGate>()};
    OpenAtScopeExit OpenGates[2] = {{Gates[0]}, {Gates[1]}};
    CompleteAllAtScopeExit CompleteAll;
    setKernelsStayPending(true);
    sycl::event E = syclex::make_event(Ctx);

    sycl::event H1 = blockQueue(Q, Gates[0]);
    sycl::event H2 = blockQueue(Q, Gates[1]);
    sycl::event P1 = Q.submit([&](sycl::handler &CGH) {
      CGH.depends_on(H1);
      CGH.single_task<BindingTestKernel>([]() {});
    });
    sycl::event P2 = Q.memcpy(Dst, Src, Count * sizeof(int), H2);

    // Held until both producers are in the backend.
    syclex::enqueue_signal_event(Q, E);
    std::shared_ptr<sycl::detail::event_binding> S1 = bindingOf(E);
    ASSERT_EQ(S1->getHandle(), nullptr);
    ASSERT_NE(S1->MCommand, nullptr);
    auto Covers = [&](const sycl::event &Producer) {
      const sycl::detail::event_impl *Impl =
          sycl::detail::getSyclObjImpl(Producer).get();
      auto IsProducer = [&](const auto &Dep) {
        return Dep.Event.get() == Impl;
      };
      return std::any_of(S1->MPreparedDepsEvents.begin(),
                         S1->MPreparedDepsEvents.end(), IsProducer) ||
             std::any_of(S1->MPreparedHostDepsEvents.begin(),
                         S1->MPreparedHostDepsEvents.end(), IsProducer);
    };
    EXPECT_TRUE(Covers(P1));
    EXPECT_TRUE(Covers(P2));

    sycl::event K = Q.submit([&](sycl::handler &CGH) {
      CGH.single_task<BindingTestKernel>([]() {});
    });
    syclex::single_task<BindingTestKernel>(Q, []() {});
    syclex::enqueue_wait_events(Waits, {P1});

    signalPending(SignalQueue, E);
    const ur_event_handle_t Signal = handleOf(E);
    ASSERT_NE(bindingOf(E), S1);
    ASSERT_NE(Signal, nullptr);
    complete(Signal);
    EXPECT_EQ(kernelLaunches(), 0u);

    // Only the kernel producer is released, and completes.
    Gates[0]->open();
    ASSERT_TRUE(eventually([&] { return KernelLaunchWaitLists.size() == 1; }));
    const ur_event_handle_t P1Handle = handleOf(P1);
    {
      std::lock_guard<std::mutex> Lock(BackendMutex);
      EXPECT_EQ(P1Handle, KernelEvents[0]);
    }
    // The wait for the kernel producer is not held behind the other one.
    EXPECT_TRUE(eventually([&] {
      return std::find(BarrierWaitLists.begin(), BarrierWaitLists.end(),
                       std::vector<ur_event_handle_t>{P1Handle}) !=
             BarrierWaitLists.end();
    }));
    complete(P1Handle);

    // The original signal and its followers still wait for the copy.
    EXPECT_EQ(S1->getHandle(), nullptr);
    EXPECT_FALSE(S1->isCompleted());
    EXPECT_EQ(handleOf(K), nullptr);
    EXPECT_EQ(kernelLaunches(), 1u);
    {
      std::lock_guard<std::mutex> Lock(BackendMutex);
      EXPECT_TRUE(MemoryEvents.empty());
    }

    Gates[1]->open();
    ASSERT_TRUE(eventually([&] {
      return KernelLaunchWaitLists.size() == 3 && S1->getHandle() != nullptr;
    }));
    const ur_event_handle_t S1Handle = S1->getHandle();
    EXPECT_NE(S1Handle, Signal);
    {
      std::lock_guard<std::mutex> Lock(BackendMutex);
      ASSERT_EQ(MemoryEvents.size(), 1u);
      const ur_event_handle_t P2Handle = MemoryEvents[0];
      // The original signal waits for the copy.
      EXPECT_TRUE(
          std::any_of(BarrierWaitLists.begin(), BarrierWaitLists.end(),
                      [&](const std::vector<ur_event_handle_t> &WaitList) {
                        return std::find(WaitList.begin(), WaitList.end(),
                                         P2Handle) != WaitList.end();
                      }));
      // Both followers wait for the original signal.
      for (size_t I = 1; I < KernelLaunchWaitLists.size(); ++I)
        EXPECT_EQ(KernelLaunchWaitLists[I],
                  std::vector<ur_event_handle_t>{S1Handle});
    }
    EXPECT_EQ(handleOf(E), Signal);

    completeAll();
    Q.wait();
    Waits.wait();
    SignalQueue.wait();
  }
  sycl::free(Src, Ctx);
  sycl::free(Dst, Ctx);
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

// Where the event of the get_wait_list tests comes from.
enum class WaitListEvent { Default, Made, Returned };

// tests-10-02 U26. E's first signal S1 is pending in the backend, and a
// consumer depends on it: held behind a host task (HeldConsumer), or through
// the scheduler bypass. E is signaled again, as S2. The event reports the
// wait list of its latest signal, while the consumer still waits for S1. The
// application then lets go of the event, the queues and the context while S1,
// S2 and the consumer are pending; everything is reclaimed once they are done.
void waitListAcrossResignal(const sycl::device &Dev, WaitListEvent Source,
                            bool HeldConsumer) {
  std::weak_ptr<sycl::detail::event_impl> WeakEvent;
  std::weak_ptr<sycl::detail::event_binding> WeakS1;
  std::vector<std::weak_ptr<sycl::detail::queue_impl>> WeakQueues;
  std::vector<ur_event_handle_t> Handles;
  ur_event_handle_t S1Handle = nullptr;
  const size_t FirstLaunch = kernelLaunches();
  const size_t ProducerLaunches = Source == WaitListEvent::Returned ? 1 : 0;
  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};
  CompleteAllAtScopeExit CompleteAll;
  setKernelsStayPending(true);
  sycl::event Consumer;
  {
    sycl::context C = Source == WaitListEvent::Default
                          ? sycl::queue{Dev}.get_context()
                          : sycl::context{Dev};
    sycl::queue Producer{C, Dev};    // out-of-order: through the bypass
    sycl::queue SignalQueue{C, Dev}; // the same
    sycl::queue Consumers =
        HeldConsumer ? sycl::queue{C, Dev, sycl::property::queue::in_order{}}
                     : sycl::queue{C, Dev};
    WeakQueues = {sycl::detail::getSyclObjImpl(Producer),
                  sycl::detail::getSyclObjImpl(SignalQueue),
                  sycl::detail::getSyclObjImpl(Consumers)};

    sycl::event E;
    if (Source == WaitListEvent::Returned) {
      sycl::event X = syclex::make_event(C);
      signalPending(Producer, X);
      Handles.push_back(handleOf(X));
      E = Producer.submit([&](sycl::handler &CGH) {
        CGH.depends_on(X);
        CGH.single_task<BindingTestKernel>([]() {});
      });
      const std::vector<sycl::event> Deps = E.get_wait_list();
      ASSERT_EQ(Deps.size(), 1u);
      EXPECT_EQ(Deps[0], X);
    } else {
      if (Source == WaitListEvent::Made)
        E = syclex::make_event(C);
      signalPending(Producer, E);
      EXPECT_TRUE(E.get_wait_list().empty());
    }
    WeakEvent = sycl::detail::getSyclObjImpl(E);
    WeakS1 = bindingOf(E);
    S1Handle = handleOf(E);
    ASSERT_NE(S1Handle, nullptr);
    Handles.push_back(S1Handle);

    if (HeldConsumer)
      blockQueue(Consumers, Gate);
    Consumer = Consumers.submit([&](sycl::handler &CGH) {
      CGH.depends_on(E);
      CGH.single_task<BindingTestKernel>([]() {});
    });
    EXPECT_TRUE(contains(Consumer.get_wait_list(), E));

    signalPending(SignalQueue, E);
    const ur_event_handle_t S2Handle = handleOf(E);
    ASSERT_NE(S2Handle, nullptr);
    Handles.push_back(S2Handle);
    // The latest signal has nothing to wait for. The consumer still reports
    // the event it depends on, which is now the new signal.
    EXPECT_TRUE(E.get_wait_list().empty());
    EXPECT_TRUE(contains(Consumer.get_wait_list(), E));
    if (HeldConsumer) {
      EXPECT_FALSE(WeakS1.expired());
      EXPECT_NE(bindingOf(E), WeakS1.lock());
    }
  }

  if (HeldConsumer) {
    // The held consumer owns S1, with what it takes to wait for it and
    // release it.
    std::shared_ptr<sycl::detail::event_binding> S1 = WeakS1.lock();
    ASSERT_NE(S1, nullptr);
    EXPECT_NE(S1->MAdapter, nullptr);
    EXPECT_EQ(S1->getHandle(), S1Handle);
    EXPECT_EQ(releases(S1Handle), 0);
    EXPECT_EQ(kernelLaunches(), FirstLaunch + ProducerLaunches);
    S1.reset();
    Gate->open();
  }
  ASSERT_TRUE(eventually([&] {
    return KernelLaunchWaitLists.size() == FirstLaunch + ProducerLaunches + 1;
  }));
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    EXPECT_EQ(KernelLaunchWaitLists.back(),
              std::vector<ur_event_handle_t>{S1Handle});
  }

  // The consumer's queue is gone.
  completeAll();
  Consumer.wait();
  EXPECT_TRUE(isComplete(Consumer));
  Consumer = sycl::event{};

  EXPECT_TRUE(eventually([&] {
    return WeakEvent.expired() && WeakS1.expired() &&
           std::all_of(WeakQueues.begin(), WeakQueues.end(),
                       [](const auto &Q) { return Q.expired(); }) &&
           std::all_of(Handles.begin(), Handles.end(), ownershipBalancedLocked);
  }));
}

class ReusableEventsWaitListTest
    : public ReusableEventsTest,
      public ::testing::WithParamInterface<std::tuple<WaitListEvent, bool>> {};

// tests-10-02 U26: an event made by make_event, and one a kernel returned.
TEST_P(ReusableEventsWaitListTest, WaitListAcrossResignalAndRelease) {
  waitListAcrossResignal(Dev, std::get<0>(GetParam()), std::get<1>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    ReusableEventsBinding, ReusableEventsWaitListTest,
    ::testing::Combine(::testing::Values(WaitListEvent::Made,
                                         WaitListEvent::Returned),
                       ::testing::Bool()),
    [](const ::testing::TestParamInfo<std::tuple<WaitListEvent, bool>> &Info) {
      return std::string(std::get<0>(Info.param) == WaitListEvent::Made
                             ? "MadeEvent"
                             : "ReturnedEvent") +
             (std::get<1>(Info.param) ? "HeldConsumer" : "BypassConsumer");
    });

// tests-10-02 U26: a default constructed event.
TEST_F(ReusableEventsBindingTest,
       DefaultEventWaitListAcrossResignalAndRelease) {
  waitListAcrossResignal(Dev, WaitListEvent::Default, /*HeldConsumer=*/true);
  waitListAcrossResignal(Dev, WaitListEvent::Default, /*HeldConsumer=*/false);
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

// A signal may be held in the runtime behind a host task. It is then a barrier
// command producing the event's new binding; the backend event is created only
// when the command is enqueued. A dependent captures that binding and waits.
TEST_F(ReusableEventsBindingTest, SignalBehindHostTaskIsDeferred) {
  sycl::queue Q = inOrderQueue();
  sycl::queue Q2 = inOrderQueue();
  sycl::event E = syclex::make_event(Ctx);

  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};
  blockQueue(Q, Gate);

  EXPECT_NO_THROW(syclex::enqueue_signal_event(Q, E));
  EXPECT_TRUE(CreatedEvents.empty());
  EXPECT_EQ(handleOf(E), nullptr);

  // Depends on the pending signal: held until it is in the backend.
  Q2.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    CGH.single_task<BindingTestKernel>([]() {});
  });
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    EXPECT_TRUE(KernelLaunchWaitLists.empty());
  }

  Gate->open();
  Q.wait();
  Q2.wait();

  std::lock_guard<std::mutex> Lock(BackendMutex);
  ASSERT_EQ(CreatedEvents.size(), 1u);
  const ur_event_handle_t Signal = CreatedEvents[0];
  EXPECT_EQ(handleOf(E), Signal);
  ASSERT_EQ(KernelLaunchWaitLists.size(), 1u);
  EXPECT_EQ(KernelLaunchWaitLists[0], std::vector<ur_event_handle_t>{Signal});
  // The barrier was asked to signal the event created for it.
  EXPECT_NE(std::find(BarrierOutEvents.begin(), BarrierOutEvents.end(), Signal),
            BarrierOutEvents.end());
}

// Two signals of the same event, each held behind its own host task, each with
// a dependent of its own. Every dependent waits for the signal it was
// submitted with, and both signals reach the backend. The gates are opened in
// submission order.
TEST_F(ReusableEventsBindingTest, TwoSignalsBehindTwoHostTasks) {
  sycl::queue Q1 = inOrderQueue();
  sycl::queue Q2 = inOrderQueue();
  sycl::queue Q3 = inOrderQueue();
  sycl::queue Q4 = inOrderQueue();
  sycl::event E = syclex::make_event(Ctx);

  auto Gate1 = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate1{Gate1};
  auto Gate2 = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate2{Gate2};

  blockQueue(Q1, Gate1);
  syclex::enqueue_signal_event(Q1, E); // S1, pending
  Q3.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    CGH.single_task<BindingTestKernel>([]() {});
  });

  blockQueue(Q2, Gate2);
  syclex::enqueue_signal_event(Q2, E); // S2, pending
  Q4.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    CGH.single_task<BindingTestKernel>([]() {});
  });

  EXPECT_TRUE(CreatedEvents.empty());
  EXPECT_EQ(handleOf(E), nullptr);

  // The first signal is released: its dependent runs, the other one stays
  // held, and the event (which represents the second signal) has no backend
  // event yet.
  Gate1->open();
  Q1.wait();
  Q3.wait();
  ur_event_handle_t First = nullptr;
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    ASSERT_EQ(CreatedEvents.size(), 1u);
    First = CreatedEvents[0];
    ASSERT_EQ(KernelLaunchWaitLists.size(), 1u);
    EXPECT_EQ(KernelLaunchWaitLists[0], std::vector<ur_event_handle_t>{First});
  }
  EXPECT_EQ(handleOf(E), nullptr);

  Gate2->open();
  Q2.wait();
  Q4.wait();
  std::lock_guard<std::mutex> Lock(BackendMutex);
  ASSERT_EQ(CreatedEvents.size(), 2u);
  const ur_event_handle_t Second = CreatedEvents[1];
  ASSERT_EQ(KernelLaunchWaitLists.size(), 2u);
  EXPECT_EQ(KernelLaunchWaitLists[1], std::vector<ur_event_handle_t>{Second});
  EXPECT_EQ(handleOf(E), Second);
}

// A wait held behind a host task keeps waiting for the signal it was submitted
// with, although the event has been signaled again since.
TEST_F(ReusableEventsBindingTest, WaitBehindHostTaskKeepsCapturedSignal) {
  sycl::queue SignalQueue = inOrderQueue();
  sycl::queue Q = inOrderQueue();
  sycl::event E = syclex::make_event(Ctx);

  syclex::enqueue_signal_event(SignalQueue, E);
  ASSERT_EQ(CreatedEvents.size(), 1u);
  const ur_event_handle_t First = CreatedEvents[0];

  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};
  blockQueue(Q, Gate);

  EXPECT_NO_THROW(syclex::enqueue_wait_event(Q, E));

  syclex::enqueue_signal_event(SignalQueue, E);
  ASSERT_EQ(CreatedEvents.size(), 2u);

  Gate->open();
  Q.wait();
  SignalQueue.wait();

  const auto Barriers = barriersWithWaitList();
  ASSERT_EQ(Barriers.size(), 1u);
  EXPECT_EQ(Barriers[0], std::vector<ur_event_handle_t>{First});
}

// A wait for a signal which is itself still held in the runtime is held too,
// and waits for that signal once it is in the backend.
TEST_F(ReusableEventsBindingTest, WaitForPendingSignalIsDeferred) {
  sycl::queue Q1 = inOrderQueue();
  sycl::queue Q2 = inOrderQueue();
  sycl::event E = syclex::make_event(Ctx);

  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};
  blockQueue(Q1, Gate);
  syclex::enqueue_signal_event(Q1, E); // pending

  EXPECT_NO_THROW(syclex::enqueue_wait_event(Q2, E));
  EXPECT_TRUE(barriersWithWaitList().empty());
  EXPECT_TRUE(CreatedEvents.empty());

  Gate->open();
  Q1.wait();
  Q2.wait();

  ASSERT_EQ(CreatedEvents.size(), 1u);
  const ur_event_handle_t Signal = CreatedEvents[0];
  const auto Barriers = barriersWithWaitList();
  ASSERT_EQ(Barriers.size(), 1u);
  EXPECT_EQ(Barriers[0], std::vector<ur_event_handle_t>{Signal});
  EXPECT_EQ(handleOf(E), Signal);
}

// The event has a backend event from a previous signal, and nothing refers to
// that signal any more, so the next signal reuses the binding in place. If
// that signal is held behind a host task, the event must not look like it has
// the previous, completed backend event while the signal is pending.
TEST_F(ReusableEventsBindingTest, DeferredResignalHidesPreviousBackendEvent) {
  sycl::queue SignalQueue = inOrderQueue();
  sycl::queue Q = inOrderQueue();
  sycl::queue Q2 = inOrderQueue();
  sycl::event E = syclex::make_event(Ctx);

  // Through the scheduler bypass: E gets a backend event.
  syclex::enqueue_signal_event(SignalQueue, E);
  SignalQueue.wait();
  ASSERT_EQ(CreatedEvents.size(), 1u);

  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};
  blockQueue(Q, Gate);

  // Held behind the host task.
  EXPECT_NO_THROW(syclex::enqueue_signal_event(Q, E));

  // The pending signal has no backend event yet, and is not complete.
  EXPECT_EQ(handleOf(E), nullptr);
  EXPECT_NE(E.get_info<sycl::info::event::command_execution_status>(),
            sycl::info::event_command_status::complete);

  // A dependent is held until the signal is in the backend, not launched
  // waiting for the previous one.
  Q2.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    CGH.single_task<BindingTestKernel>([]() {});
  });
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    EXPECT_TRUE(KernelLaunchWaitLists.empty());
  }

  Gate->open();
  Q.wait();
  Q2.wait();

  std::lock_guard<std::mutex> Lock(BackendMutex);
  ASSERT_EQ(KernelLaunchWaitLists.size(), 1u);
  // The kernel waits for the event the second signal's barrier signaled.
  ASSERT_FALSE(BarrierOutEvents.empty());
  EXPECT_EQ(KernelLaunchWaitLists[0],
            std::vector<ur_event_handle_t>{handleOf(E)});
}

// With and without native support, and whether the test keeps the previous
// signal's binding (so that the new signal gets a binding of its own) or not
// (so that the binding is reused in place).
class ReusableEventsDeferredResignalTest
    : public ReusableEventsTest,
      public ::testing::WithParamInterface<std::tuple<bool, bool>> {
protected:
  bool nativeSupport() const override { return std::get<0>(GetParam()); }
  bool retainFirst() const { return std::get<1>(GetParam()); }
};

// tests-10-02 U04. The same, with waiters: a host wait for E and consumers on
// other queues submitted while the signal is held. Once the signal reaches
// the backend its new backend event is published, which lets the device
// consumer through, but nothing completes before that backend event does.
// The previous, completed backend event never stands in for the new signal.
TEST_P(ReusableEventsDeferredResignalTest, PendingResignalExposesNoOlderEvent) {
  sycl::queue SignalQueue = inOrderQueue();
  sycl::queue Q = inOrderQueue();
  sycl::queue Consumers = inOrderQueue();
  sycl::queue HostConsumers = inOrderQueue();
  sycl::event E = syclex::make_event(Ctx);
  std::atomic<bool> ConsumerRan{false};

  // Through the scheduler bypass, finished.
  syclex::enqueue_signal_event(SignalQueue, E);
  SignalQueue.wait();
  const ur_event_handle_t First = handleOf(E);
  ASSERT_NE(First, nullptr);
  ASSERT_TRUE(isComplete(E));
  const sycl::detail::event_binding *FirstBinding = bindingOf(E).get();
  std::shared_ptr<sycl::detail::event_binding> RetainedFirst;
  if (retainFirst())
    RetainedFirst = bindingOf(E);
  else
    ASSERT_EQ(sycl::detail::getSyclObjImpl(E)->getBinding().use_count(), 1);
  auto WaitsFor = [](ur_event_handle_t Handle) {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    return std::count(WaitedEvents.begin(), WaitedEvents.end(), Handle);
  };
  const auto FirstWaits = WaitsFor(First);

  std::future<void> Waiter;
  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};
  CompleteAllAtScopeExit CompleteAll;
  blockQueue(Q, Gate);

  // Held behind the host task.
  syclex::enqueue_signal_event(Q, E);
  std::shared_ptr<sycl::detail::event_binding> Second = bindingOf(E);
  if (retainFirst()) {
    EXPECT_NE(Second.get(), FirstBinding);
    EXPECT_EQ(releases(First), 0);
    EXPECT_EQ(RetainedFirst->getHandle(), First);
  } else {
    EXPECT_EQ(Second.get(), FirstBinding);
    // The previous backend event is given up, not kept for the new signal.
    EXPECT_EQ(releases(First), 1);
  }
  EXPECT_EQ(handleOf(E), nullptr);
  EXPECT_FALSE(isComplete(E));

  Waiter = std::async(std::launch::async, [E]() mutable { E.wait(); });
  Consumers.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    CGH.single_task<BindingTestKernel>([]() {});
  });
  HostConsumers.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    CGH.host_task([&ConsumerRan] { ConsumerRan = true; });
  });
  // Best effort: the waiter cannot be observed entering the wait.
  EXPECT_TRUE(stillBlocked(Waiter));
  EXPECT_EQ(kernelLaunches(), 0u);
  EXPECT_FALSE(ConsumerRan);

  // The signal reaches the backend; its backend event stays pending.
  setBarriersStayPending(true);
  Gate->open();
  ASSERT_TRUE(eventually([&] { return Second->getHandle() != nullptr; }));
  setBarriersStayPending(false);
  const ur_event_handle_t Signal = Second->getHandle();
  EXPECT_NE(Signal, First);
  EXPECT_EQ(handleOf(E), Signal);

  // The device consumer is let through, waiting for the new backend event.
  ASSERT_TRUE(eventually([&] { return KernelLaunchWaitLists.size() == 1; }));
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    EXPECT_EQ(KernelLaunchWaitLists[0], std::vector<ur_event_handle_t>{Signal});
  }
  // The host waiter and the host task wait for it on the host.
  EXPECT_TRUE(eventually([&] {
    return std::count(WaitedEvents.begin(), WaitedEvents.end(), Signal) >= 2;
  }));
  EXPECT_TRUE(stillBlocked(Waiter));
  EXPECT_FALSE(ConsumerRan);
  EXPECT_FALSE(isComplete(E));

  complete(Signal);
  EXPECT_TRUE(finishes(Waiter));
  EXPECT_TRUE(eventually([&] { return ConsumerRan.load(); }));
  EXPECT_TRUE(isComplete(E));
  EXPECT_EQ(handleOf(E), Signal);
  // Nobody took the previous backend event for the new signal.
  EXPECT_EQ(WaitsFor(First), FirstWaits);
  if (retainFirst()) {
    EXPECT_EQ(RetainedFirst->getHandle(), First);
    EXPECT_TRUE(RetainedFirst->isCompleted());
  }

  Q.wait();
  Consumers.wait();
  HostConsumers.wait();
}

INSTANTIATE_TEST_SUITE_P(
    ReusableEventsBinding, ReusableEventsDeferredResignalTest,
    ::testing::Combine(::testing::Bool(), ::testing::Bool()),
    [](const ::testing::TestParamInfo<std::tuple<bool, bool>> &Info) {
      return std::string(std::get<0>(Info.param) ? "NativeSupport"
                                                 : "NoNativeSupport") +
             (std::get<1>(Info.param) ? "RetainedFirst" : "ReusedFirst");
    });

// The backend event of an IPC event is shared with another process, so a
// signal of it cannot be held behind a host task: enqueue_signal_event throws,
// and the event keeps its backend event.
TEST_F(ReusableEventsBindingTest, IpcSignalBehindHostTaskThrows) {
  sycl::queue SignalQueue = inOrderQueue();
  sycl::queue Q = inOrderQueue();
  sycl::event E =
      syclex::make_event(Ctx, syclex::properties{syclex::enable_ipc{true}});

  // Through the scheduler bypass: E gets its backend event.
  syclex::enqueue_signal_event(SignalQueue, E);
  SignalQueue.wait();
  ASSERT_EQ(CreatedEvents.size(), 1u);
  const ur_event_handle_t Exported = CreatedEvents[0];
  ASSERT_EQ(handleOf(E), Exported);

  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};
  blockQueue(Q, Gate);

  try {
    syclex::enqueue_signal_event(Q, E);
    FAIL() << "a deferred signal of an IPC event did not throw";
  } catch (const sycl::exception &Ex) {
    EXPECT_EQ(Ex.code(), sycl::make_error_code(sycl::errc::invalid));
  }
  EXPECT_EQ(handleOf(E), Exported);
  EXPECT_EQ(CreatedEvents.size(), 1u);
  EXPECT_EQ(releases(Exported), 0);
}

// Neither can a wait for an IPC event be held behind a host task. A wait from
// a queue with nothing pending still works.
TEST_F(ReusableEventsBindingTest, IpcWaitBehindHostTaskThrows) {
  sycl::queue SignalQueue = inOrderQueue();
  sycl::queue Q = inOrderQueue();
  sycl::queue Q2 = inOrderQueue();
  sycl::event E =
      syclex::make_event(Ctx, syclex::properties{syclex::enable_ipc{true}});
  sycl::event Plain = syclex::make_event(Ctx);
  syclex::enqueue_signal_event(SignalQueue, E);
  syclex::enqueue_signal_event(SignalQueue, Plain);
  SignalQueue.wait();

  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};
  blockQueue(Q, Gate);

  try {
    syclex::enqueue_wait_event(Q, E);
    FAIL() << "a deferred wait for an IPC event did not throw";
  } catch (const sycl::exception &Ex) {
    EXPECT_EQ(Ex.code(), sycl::make_error_code(sycl::errc::invalid));
  }
  try {
    syclex::enqueue_wait_events(Q, {Plain, E});
    FAIL() << "a deferred wait for a list with an IPC event did not throw";
  } catch (const sycl::exception &Ex) {
    EXPECT_EQ(Ex.code(), sycl::make_error_code(sycl::errc::invalid));
  }
  // Without the IPC event the wait is simply held.
  EXPECT_NO_THROW(syclex::enqueue_wait_events(Q, {Plain}));

  EXPECT_NO_THROW(syclex::enqueue_wait_event(Q2, E));
  Gate->open();
  Q.wait();
  Q2.wait();
}

// The same holds for an event imported from another process: its backend event
// is the exporting process's signal.
TEST_F(ReusableEventsBindingTest, ImportedIpcWaitBehindHostTaskThrows) {
  sycl::queue Q = inOrderQueue();
  std::byte HandleBytes[8] = {};
  syclex::ipc::handle_data_t HandleData{HandleBytes,
                                        HandleBytes + sizeof(HandleBytes)};
  sycl::event Imported = syclex::ipc::event::open(HandleData, Ctx);
  ASSERT_NE(handleOf(Imported), nullptr);

  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};
  blockQueue(Q, Gate);

  try {
    syclex::enqueue_wait_event(Q, Imported);
    FAIL() << "a deferred wait for an imported IPC event did not throw";
  } catch (const sycl::exception &Ex) {
    EXPECT_EQ(Ex.code(), sycl::make_error_code(sycl::errc::invalid));
  }
  Gate->open();
  Q.wait();
}

// An imported event can be signaled, but only through the scheduler bypass: a
// signal held behind a host task would have to give up the imported backend
// event, which the exporting process waits for.
TEST_F(ReusableEventsBindingTest, ImportedIpcSignalBehindHostTaskThrows) {
  sycl::queue Q = inOrderQueue();
  std::byte HandleBytes[8] = {};
  syclex::ipc::handle_data_t HandleData{HandleBytes,
                                        HandleBytes + sizeof(HandleBytes)};
  sycl::event Imported = syclex::ipc::event::open(HandleData, Ctx);
  const ur_event_handle_t Shared = handleOf(Imported);
  ASSERT_NE(Shared, nullptr);

  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};
  blockQueue(Q, Gate);

  try {
    syclex::enqueue_signal_event(Q, Imported);
    FAIL() << "a deferred signal of an imported IPC event did not throw";
  } catch (const sycl::exception &Ex) {
    EXPECT_EQ(Ex.code(), sycl::make_error_code(sycl::errc::invalid));
  }
  EXPECT_EQ(handleOf(Imported), Shared);
  EXPECT_EQ(releases(Shared), 0);
  EXPECT_TRUE(CreatedEvents.empty());
  Gate->open();
  Q.wait();
}

// A signal of an imported event while a dependency still holds the previous
// signal moves the event on to a new binding. The new binding signals the
// imported backend event as well, not a new one.
TEST_F(ReusableEventsBindingTest,
       ImportedIpcResignalKeepsImportedBackendEvent) {
  sycl::queue Q = inOrderQueue();
  sycl::queue SignalQueue = inOrderQueue();
  std::byte HandleBytes[8] = {};
  syclex::ipc::handle_data_t HandleData{HandleBytes,
                                        HandleBytes + sizeof(HandleBytes)};
  sycl::event Imported = syclex::ipc::event::open(HandleData, Ctx);
  const ur_event_handle_t Shared = handleOf(Imported);
  ASSERT_NE(Shared, nullptr);

  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};
  blockQueue(Q, Gate);

  // Held behind the host task; captures the current binding.
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Imported);
    CGH.single_task<BindingTestKernel>([]() {});
  });

  // Through the scheduler bypass, onto a new binding.
  syclex::enqueue_signal_event(SignalQueue, Imported);
  EXPECT_TRUE(CreatedEvents.empty());
  EXPECT_EQ(handleOf(Imported), Shared);
  {
    std::lock_guard<std::mutex> Lock(BackendMutex);
    ASSERT_FALSE(BarrierOutEvents.empty());
    EXPECT_EQ(BarrierOutEvents.back(), Shared);
    // One reference for each binding.
    EXPECT_EQ(RetainCounts[Shared], 1);
  }
  EXPECT_EQ(releases(Shared), 0);

  Gate->open();
  Q.wait();
  SignalQueue.wait();

  std::lock_guard<std::mutex> Lock(BackendMutex);
  ASSERT_EQ(KernelLaunchWaitLists.size(), 1u);
  EXPECT_EQ(KernelLaunchWaitLists[0], std::vector<ur_event_handle_t>{Shared});
  EXPECT_TRUE(CreatedEvents.empty());
}

} // anonymous namespace
