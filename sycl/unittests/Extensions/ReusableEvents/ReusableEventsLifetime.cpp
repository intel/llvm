//==-- ReusableEventsLifetime.cpp --- Lifetime of the signals of an event --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Every signal of a reusable event has an event_binding of its own, which
// owns the signal's backend event. The tests below follow the bindings and the
// backend events of several signals of one event through their consumers, and
// check that each of them belongs to the right signal and lives exactly as
// long as something refers to it.
//
// The scenario numbers refer to the reusable events test plan
// (tests-10-02.md); the findings to review-10-01.md and review-10-02.md.
//
//===----------------------------------------------------------------------===//

#include "ReusableEventsFixture.hpp"

#include <atomic>
#include <functional>

namespace {

using namespace reusable_events_test;

using ReusableEventsLifetimeTest = ReusableEventsTest;
using ReusableEventsLifetimeSupportTest = ReusableEventsSupportTest;

INSTANTIATE_TEST_SUITE_P(ReusableEventsLifetime,
                         ReusableEventsLifetimeSupportTest, ::testing::Bool(),
                         supportName);

// Signals E on Q; the backend event of the signal stays pending until the test
// completes it.
void signalPending(sycl::queue &Q, sycl::event &E) {
  setBarriersStayPending(true);
  syclex::enqueue_signal_event(Q, E);
  setBarriersStayPending(false);
}

struct TwoSignals {
  ur_event_handle_t First = nullptr;
  ur_event_handle_t Second = nullptr;
};

// Signals E twice on Q, each time with a pending backend event. A kernel held
// behind a host task captures the first signal, so the second one gets a
// binding of its own. Checks that each binding has the adapter to wait for,
// query and release its backend event, and that the signals complete
// independently.
void signalCapturedTwice(sycl::queue &Q, sycl::event &E, TwoSignals &Result) {
  sycl::queue Other{Q.get_context(), Q.get_device(),
                    sycl::property::queue::in_order{}};
  EXPECT_TRUE(isComplete(E));

  signalPending(Q, E);
  std::shared_ptr<sycl::detail::event_binding> First = bindingOf(E);
  Result.First = handleOf(E);
  ASSERT_NE(Result.First, nullptr);
  EXPECT_NE(First->MAdapter, nullptr);
  EXPECT_FALSE(isComplete(E));

  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};
  blockQueue(Other, Gate);
  Other.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    CGH.single_task<BindingTestKernel>([]() {});
  });

  signalPending(Q, E);
  std::shared_ptr<sycl::detail::event_binding> Second = bindingOf(E);
  Result.Second = handleOf(E);
  ASSERT_NE(First, Second);
  ASSERT_NE(Result.Second, nullptr);
  EXPECT_NE(Result.Second, Result.First);
  EXPECT_NE(Second->MAdapter, nullptr);

  // The signals complete independently.
  complete(Result.Second);
  EXPECT_TRUE(isComplete(E));
  EXPECT_FALSE(First->isCompleted());

  Gate->open();
  complete(Result.First);
  Other.wait();
  Q.wait();
  E.wait();
  EXPECT_TRUE(First->isCompleted());

  std::lock_guard<std::mutex> Lock(BackendMutex);
  ASSERT_EQ(KernelLaunchWaitLists.size(), 1u);
  EXPECT_EQ(KernelLaunchWaitLists[0],
            std::vector<ur_event_handle_t>{Result.First});
}

// tests-10-02 U01, control: an event made for the default context.
TEST_P(ReusableEventsLifetimeSupportTest, MadeEventSignalsOwnTheirHandles) {
  TwoSignals Signals;
  {
    sycl::queue Q{Dev, sycl::property::queue::in_order{}};
    sycl::event E = syclex::make_event(Q.get_context());
    signalCapturedTwice(Q, E, Signals);
  }
  ASSERT_NE(Signals.First, nullptr);
  ASSERT_NE(Signals.Second, nullptr);
  EXPECT_TRUE(eventually([&] {
    return ownershipBalancedLocked(Signals.First) &&
           ownershipBalancedLocked(Signals.Second);
  }));
}

// tests-10-02 U01: a default constructed event gets its context, the default
// one, when it is first signaled.
// Known defect: review-10-02 #2 (lazy context initialization leaves the
// binding's adapter null; waiting for or releasing the backend event asserts).
TEST_P(ReusableEventsLifetimeSupportTest,
       DISABLED_DefaultConstructedEventSignalsOwnTheirHandles) {
  TwoSignals Signals;
  {
    sycl::queue Q{Dev, sycl::property::queue::in_order{}};
    sycl::event E;
    signalCapturedTwice(Q, E, Signals);
  }
  ASSERT_NE(Signals.First, nullptr);
  ASSERT_NE(Signals.Second, nullptr);
  EXPECT_TRUE(eventually([&] {
    return ownershipBalancedLocked(Signals.First) &&
           ownershipBalancedLocked(Signals.Second);
  }));
}

// tests-10-02 U02. Copies and moves of an event are the same event: they all
// follow its latest signal. A consumer keeps the signal it was submitted with.
TEST_F(ReusableEventsLifetimeTest,
       AliasesFollowTheEventConsumersKeepTheSignal) {
  ur_event_handle_t First = nullptr;
  ur_event_handle_t Second = nullptr;
  std::atomic<bool> ConsumerRan{false};
  {
    sycl::queue SignalQueue = inOrderQueue();
    sycl::queue Q = inOrderQueue();
    sycl::event Consumer;
    {
      sycl::event E = syclex::make_event(Ctx);
      sycl::event A = E;
      sycl::event Copy = E;
      sycl::event B = std::move(Copy);

      signalPending(SignalQueue, E);
      First = handleOf(E);
      ASSERT_NE(First, nullptr);

      // The host task waits for its dependency on the host.
      Consumer = Q.submit([&](sycl::handler &CGH) {
        CGH.depends_on(A);
        CGH.host_task([&] { ConsumerRan = true; });
      });

      signalPending(SignalQueue, B);
      Second = handleOf(B);
      ASSERT_NE(Second, nullptr);
      ASSERT_NE(Second, First);
      for (const sycl::event *Alias : {&E, &A, &B}) {
        EXPECT_EQ(bindingOf(*Alias), bindingOf(B));
        EXPECT_EQ(handleOf(*Alias), Second);
      }
      EXPECT_TRUE(E == A && A == B);

      complete(Second);
      for (const sycl::event *Alias : {&E, &A, &B})
        EXPECT_TRUE(isComplete(*Alias));
      EXPECT_TRUE(eventually([&] {
        return std::count(WaitedEvents.begin(), WaitedEvents.end(), First) != 0;
      }));
      EXPECT_FALSE(ConsumerRan);
      EXPECT_FALSE(isComplete(Consumer));
    }
    // The aliases are gone; the consumer still needs the first signal.
    EXPECT_EQ(releases(First), 0);

    complete(First);
    Consumer.wait();
    EXPECT_TRUE(ConsumerRan);
    EXPECT_FALSE(waitedFor(Second));
    Consumer = sycl::event{};
    Q.wait();
    SignalQueue.wait();
  }
  EXPECT_TRUE(eventually([&] {
    return ownershipBalancedLocked(First) && ownershipBalancedLocked(Second);
  }));
}

// What the producers of tests-10-02 U06 operate on.
struct ProducerResources {
  explicit ProducerResources(sycl::queue &Q)
      : Usm{sycl::malloc_device<int>(Size, Q)},
        Usm2{sycl::malloc_device<int>(Size, Q)}, Q{Q} {}
  ~ProducerResources() {
    sycl::free(Usm, Q);
    sycl::free(Usm2, Q);
  }

  static constexpr size_t Size = 4;
  sycl::buffer<int, 1> Buf{sycl::range<1>{Size}};
  sycl::buffer<int, 1> Buf2{sycl::range<1>{Size}};
  int Host[Size] = {};
  int *Usm;
  int *Usm2;
  sycl::queue &Q;
};

struct Producer {
  const char *Name;
  std::function<sycl::event(sycl::queue &, ProducerResources &)> Submit;
};

class ReusableEventsProducerTest
    : public ReusableEventsTest,
      public ::testing::WithParamInterface<Producer> {};

// tests-10-02 U06. The event returned by a submission held in the runtime is
// signaled elsewhere, and the new signal completes first. The operation, and
// a consumer submitted before the signal, keep the original signal: the
// operation produces its backend event, which the consumer waits for.
TEST_P(ReusableEventsProducerTest, ResignaledProducerEventKeepsItsOperation) {
  sycl::queue Q = inOrderQueue();
  sycl::queue Consumers = inOrderQueue();
  sycl::queue SignalQueue = inOrderQueue();
  ProducerResources Resources{Q};
  {
    auto Gate = std::make_shared<HostTaskGate>();
    OpenAtScopeExit OpenGate{Gate};
    blockQueue(Q, Gate);

    sycl::event E = GetParam().Submit(Q, Resources);
    std::shared_ptr<sycl::detail::event_binding> Original = bindingOf(E);
    EXPECT_EQ(Original->getHandle(), nullptr);

    Consumers.submit([&](sycl::handler &CGH) {
      CGH.depends_on(E);
      CGH.single_task<BindingTestKernel>([]() {});
    });

    syclex::enqueue_signal_event(SignalQueue, E);
    SignalQueue.wait();
    const ur_event_handle_t Signal = handleOf(E);
    EXPECT_NE(bindingOf(E), Original);
    EXPECT_TRUE(isComplete(E));

    // The operation and its consumer are still held.
    EXPECT_EQ(Original->getHandle(), nullptr);
    EXPECT_FALSE(Original->isCompleted());
    EXPECT_EQ(kernelLaunches(), 0u);

    Gate->open();
    Q.wait();
    Consumers.wait();

    const ur_event_handle_t OriginalHandle = Original->getHandle();
    EXPECT_NE(OriginalHandle, Signal);
    EXPECT_TRUE(Original->isCompleted());
    EXPECT_EQ(handleOf(E), Signal);

    std::lock_guard<std::mutex> Lock(BackendMutex);
    ASSERT_FALSE(KernelLaunchWaitLists.empty());
    // An operation without a backend event leaves nothing to wait for.
    const std::vector<ur_event_handle_t> Expected =
        OriginalHandle ? std::vector<ur_event_handle_t>{OriginalHandle}
                       : std::vector<ur_event_handle_t>{};
    EXPECT_EQ(KernelLaunchWaitLists.back(), Expected);
  }
}

// Device globals and native commands are not covered: the mock has no device
// image with a device global, nor a native-command backend.
INSTANTIATE_TEST_SUITE_P(
    ReusableEventsLifetime, ReusableEventsProducerTest,
    ::testing::Values(
        Producer{"Kernel",
                 [](sycl::queue &Q, ProducerResources &) {
                   return Q.submit([&](sycl::handler &CGH) {
                     CGH.single_task<BindingTestKernel>([]() {});
                   });
                 }},
        Producer{"BufferFill",
                 [](sycl::queue &Q, ProducerResources &R) {
                   return Q.submit([&](sycl::handler &CGH) {
                     sycl::accessor A{R.Buf, CGH, sycl::write_only};
                     CGH.fill(A, 1);
                   });
                 }},
        Producer{"CopyAccessorToPointer",
                 [](sycl::queue &Q, ProducerResources &R) {
                   return Q.submit([&](sycl::handler &CGH) {
                     sycl::accessor A{R.Buf, CGH, sycl::read_only};
                     CGH.copy(A, R.Host);
                   });
                 }},
        Producer{"CopyPointerToAccessor",
                 [](sycl::queue &Q, ProducerResources &R) {
                   return Q.submit([&](sycl::handler &CGH) {
                     sycl::accessor A{R.Buf, CGH, sycl::write_only};
                     CGH.copy(static_cast<const int *>(R.Host), A);
                   });
                 }},
        Producer{"CopyAccessorToAccessor",
                 [](sycl::queue &Q, ProducerResources &R) {
                   return Q.submit([&](sycl::handler &CGH) {
                     sycl::accessor Src{R.Buf, CGH, sycl::read_only};
                     sycl::accessor Dst{R.Buf2, CGH, sycl::write_only};
                     CGH.copy(Src, Dst);
                   });
                 }},
        Producer{"UsmCopy",
                 [](sycl::queue &Q, ProducerResources &R) {
                   return Q.memcpy(R.Usm2, R.Usm,
                                   ProducerResources::Size * sizeof(int));
                 }},
        Producer{"UsmFill",
                 [](sycl::queue &Q, ProducerResources &R) {
                   return Q.fill(R.Usm, 1, ProducerResources::Size);
                 }},
        Producer{"Prefetch",
                 [](sycl::queue &Q, ProducerResources &R) {
                   return Q.prefetch(R.Usm,
                                     ProducerResources::Size * sizeof(int));
                 }},
        Producer{"MemAdvise",
                 [](sycl::queue &Q, ProducerResources &R) {
                   return Q.mem_advise(
                       R.Usm, ProducerResources::Size * sizeof(int), 0);
                 }},
        Producer{"EmptyCommandGroup",
                 [](sycl::queue &Q, ProducerResources &) {
                   return Q.submit([](sycl::handler &) {});
                 }}),
    [](const ::testing::TestParamInfo<Producer> &Info) {
      return std::string(Info.param.Name);
    });

// tests-10-02 U07. Three signals of one event, each held behind a host task
// of its own, each captured by a kernel, a USM copy, a host task and a queue
// wait. The backend events complete in the order third, first, second; every
// group follows its own signal.
TEST_F(ReusableEventsLifetimeTest, ConsumerGroupsFollowTheirOwnSignal) {
  constexpr int Signals = 3;
  std::weak_ptr<sycl::detail::event_binding> Bindings[Signals];
  ur_event_handle_t Handles[Signals] = {};
  std::atomic<bool> HostTaskRan[Signals] = {};
  int *Src = sycl::malloc_device<int>(1, Dev, Ctx);
  int *Dst = sycl::malloc_device<int>(1, Dev, Ctx);
  {
    sycl::queue SignalQueues[Signals] = {inOrderQueue(), inOrderQueue(),
                                         inOrderQueue()};
    // Out-of-order, one per group: the queue wait orders only its own group.
    sycl::queue Groups[Signals] = {sycl::queue{Ctx, Dev}, sycl::queue{Ctx, Dev},
                                   sycl::queue{Ctx, Dev}};
    std::shared_ptr<HostTaskGate> Gates[Signals] = {
        std::make_shared<HostTaskGate>(), std::make_shared<HostTaskGate>(),
        std::make_shared<HostTaskGate>()};
    // Constructed in place: a copied guard would open its gate on destruction.
    OpenAtScopeExit OpenGates[Signals] = {{Gates[0]}, {Gates[1]}, {Gates[2]}};
    CompleteAllAtScopeExit CompleteAll;
    setBarriersStayPending(true);
    {
      sycl::event E = syclex::make_event(Ctx);
      for (int I = 0; I < Signals; ++I) {
        blockQueue(SignalQueues[I], Gates[I]);
        syclex::enqueue_signal_event(SignalQueues[I], E);
        Bindings[I] = bindingOf(E);

        sycl::queue &G = Groups[I];
        G.submit([&](sycl::handler &CGH) {
          CGH.depends_on(E);
          CGH.single_task<BindingTestKernel>([]() {});
        });
        G.memcpy(Dst, Src, sizeof(int), E);
        G.submit([&](sycl::handler &CGH) {
          CGH.depends_on(E);
          CGH.host_task([&HostTaskRan, I] { HostTaskRan[I] = true; });
        });
        syclex::enqueue_wait_event(G, E);
      }
      EXPECT_EQ(kernelLaunches(), 0u);
      // The consumers hold the signals; the event is not needed any more.
    }

    // Each signal reaches the backend, and its group with it.
    for (int I = 0; I < Signals; ++I) {
      Gates[I]->open();
      ASSERT_TRUE(eventually([&] {
        return KernelLaunchWaitLists.size() == size_t(I + 1) &&
               MemoryWaitLists.size() == size_t(I + 1);
      }));
      auto Binding = Bindings[I].lock();
      ASSERT_TRUE(Binding);
      Handles[I] = Binding->getHandle();
      ASSERT_NE(Handles[I], nullptr);
      std::lock_guard<std::mutex> Lock(BackendMutex);
      EXPECT_EQ(KernelLaunchWaitLists[I],
                std::vector<ur_event_handle_t>{Handles[I]});
      EXPECT_EQ(MemoryWaitLists[I], std::vector<ur_event_handle_t>{Handles[I]});
    }
    EXPECT_TRUE(eventually([&] {
      return std::all_of(Handles, Handles + Signals, [](ur_event_handle_t H) {
        return std::count(WaitedEvents.begin(), WaitedEvents.end(), H) != 0;
      });
    }));
    const std::set<ur_event_handle_t> Distinct(Handles, Handles + Signals);
    EXPECT_EQ(Distinct.size(), size_t(Signals));

    complete(Handles[2]);
    EXPECT_TRUE(eventually([&] { return HostTaskRan[2].load(); }));
    EXPECT_FALSE(HostTaskRan[0]);
    EXPECT_FALSE(HostTaskRan[1]);
    complete(Handles[0]);
    EXPECT_TRUE(eventually([&] { return HostTaskRan[0].load(); }));
    EXPECT_FALSE(HostTaskRan[1]);
    complete(Handles[1]);
    EXPECT_TRUE(eventually([&] { return HostTaskRan[1].load(); }));

    // The queue waits waited for their own group's signal.
    for (int I = 0; I < Signals; ++I) {
      const auto Barriers = barriersWithWaitList();
      EXPECT_EQ(std::count(Barriers.begin(), Barriers.end(),
                           std::vector<ur_event_handle_t>{Handles[I]}),
                1);
    }

    completeAll();
    for (sycl::queue &G : Groups)
      G.wait();
    for (sycl::queue &Q : SignalQueues)
      Q.wait();
  }
  sycl::free(Src, Ctx);
  sycl::free(Dst, Ctx);
  EXPECT_TRUE(eventually([&] {
    return std::all_of(Bindings, Bindings + Signals,
                       [](const auto &B) { return B.expired(); }) &&
           std::all_of(Handles, Handles + Signals, ownershipBalancedLocked);
  }));
}

// tests-10-02 U23. Signals the same event repeatedly behind a host task on an
// in-order queue: every signal after the first depends on the previous one.
// Once the application and the queue let go of everything, the event, its
// bindings and their backend events are all released. With Consume, a kernel
// on another queue depends on every deferred signal.
void signalRepeatedlyAndRelease(int DeferredSignals, bool Consume,
                                const sycl::context &Ctx) {
  std::weak_ptr<sycl::detail::event_impl> WeakEvent;
  std::vector<std::weak_ptr<sycl::detail::event_binding>> WeakBindings;
  std::vector<ur_event_handle_t> Handles;
  // Keeps the deferred signals' bindings until their handles are read; it is
  // emptied before anything is expected to be released.
  std::vector<std::shared_ptr<sycl::detail::event_binding>> Deferred;
  // Launches recorded by an earlier call in the same test.
  const size_t FirstLaunch = kernelLaunches();
  {
    sycl::device Dev = Ctx.get_devices()[0];
    sycl::queue Q{Ctx, Dev, sycl::property::queue::in_order{}};
    sycl::queue Consumers{Ctx, Dev, sycl::property::queue::in_order{}};
    auto Gate = std::make_shared<HostTaskGate>();
    OpenAtScopeExit OpenGate{Gate};
    {
      sycl::event E = syclex::make_event(Ctx);
      WeakEvent = sycl::detail::getSyclObjImpl(E);

      // Through the scheduler bypass first.
      syclex::enqueue_signal_event(Q, E);
      WeakBindings.push_back(bindingOf(E));
      Handles.push_back(handleOf(E));

      blockQueue(Q, Gate);
      for (int I = 0; I < DeferredSignals; ++I) {
        syclex::enqueue_signal_event(Q, E);
        WeakBindings.push_back(bindingOf(E));
        Deferred.push_back(bindingOf(E));
        if (Consume) {
          Consumers.submit([&](sycl::handler &CGH) {
            CGH.depends_on(E);
            CGH.single_task<BindingTestKernel>([]() {});
          });
        }
      }
    }
    Gate->open();
    Q.wait();
    Consumers.wait();
    for (const auto &Binding : Deferred)
      Handles.push_back(Binding->getHandle());
    Deferred.clear();
    // Every consumer waited for its own signal.
    std::lock_guard<std::mutex> Lock(BackendMutex);
    ASSERT_EQ(KernelLaunchWaitLists.size(),
              FirstLaunch + (Consume ? DeferredSignals : 0));
    for (int I = 0; Consume && I < DeferredSignals; ++I)
      EXPECT_EQ(KernelLaunchWaitLists[FirstLaunch + I],
                std::vector<ur_event_handle_t>{Handles[I + 1]});
  }
  ASSERT_EQ(Handles.size(), WeakBindings.size());
  EXPECT_TRUE(eventually([&] { return WeakEvent.expired(); }))
      << "the event outlived its owners";
  for (size_t I = 0; I < WeakBindings.size(); ++I)
    EXPECT_TRUE(WeakBindings[I].expired()) << "signal " << I << " leaked";
  for (ur_event_handle_t Handle : Handles) {
    if (Handle) {
      EXPECT_TRUE(ownershipBalanced(Handle));
    }
  }
}

// tests-10-02 U23, control: a single signal behind the host task.
TEST_P(ReusableEventsLifetimeSupportTest, DeferredSignalIsReleased) {
  signalRepeatedlyAndRelease(1, /*Consume=*/true, Ctx);
  signalRepeatedlyAndRelease(1, /*Consume=*/false, Ctx);
}

// tests-10-02 U23, control: a consumer of the last signal breaks the cycle
// below when its command is destroyed, by clearing the dependencies of the
// signal it depends on.
TEST_P(ReusableEventsLifetimeSupportTest,
       RepeatedConsumedDeferredSignalsAreReleased) {
  signalRepeatedlyAndRelease(2, /*Consume=*/true, Ctx);
  signalRepeatedlyAndRelease(3, /*Consume=*/true, Ctx);
}

// tests-10-02 U23.
// Known defect: review-10-02 #8 (a signal depending on the previous signal of
// the same event owns the event through its captured dependency, a cycle).
TEST_P(ReusableEventsLifetimeSupportTest,
       DISABLED_RepeatedDeferredSignalsAreReleased) {
  signalRepeatedlyAndRelease(2, /*Consume=*/false, Ctx);
  signalRepeatedlyAndRelease(3, /*Consume=*/false, Ctx);
}

class ReusableEventsCleanupOrderTest
    : public ReusableEventsTest,
      public ::testing::WithParamInterface<bool> {};

// tests-10-02 U25. The producer of an event and a signal of the event are both
// held behind host tasks of their own. Whichever reaches the backend and is
// cleaned up first, it leaves the other signal's state alone. A consumer of
// the second signal waits for it.
TEST_P(ReusableEventsCleanupOrderTest, CleanupOfOneSignalLeavesTheOtherAlone) {
  const bool OlderFirst = GetParam();
  sycl::queue Q1 = inOrderQueue();
  sycl::queue Q2 = inOrderQueue();
  sycl::queue Q3 = inOrderQueue();
  auto Gate1 = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate1{Gate1};
  auto Gate2 = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate2{Gate2};
  blockQueue(Q1, Gate1);
  blockQueue(Q2, Gate2);

  sycl::event E = Q1.submit(
      [&](sycl::handler &CGH) { CGH.single_task<BindingTestKernel>([]() {}); });
  std::shared_ptr<sycl::detail::event_binding> First = bindingOf(E);
  syclex::enqueue_signal_event(Q2, E);
  std::shared_ptr<sycl::detail::event_binding> Second = bindingOf(E);
  ASSERT_NE(First, Second);
  EXPECT_NE(First->MCommand, nullptr);
  EXPECT_NE(Second->MCommand, nullptr);
  Q3.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    CGH.single_task<BindingTestKernel>([]() {});
  });

  auto CheckFirstDone = [&] {
    EXPECT_NE(First->getHandle(), nullptr);
    EXPECT_TRUE(First->isCompleted());
  };
  if (OlderFirst) {
    Gate1->open();
    Q1.wait();
    CheckFirstDone();
    // The second signal is still pending, and the event with it.
    EXPECT_EQ(Second->getHandle(), nullptr);
    EXPECT_NE(Second->MCommand, nullptr);
    EXPECT_FALSE(Second->isCompleted());
    EXPECT_EQ(handleOf(E), nullptr);
    EXPECT_FALSE(isComplete(E));
    Gate2->open();
    Q2.wait();
  } else {
    Gate2->open();
    Q2.wait();
    // The first signal is still pending.
    EXPECT_EQ(First->getHandle(), nullptr);
    EXPECT_NE(First->MCommand, nullptr);
    EXPECT_FALSE(First->isCompleted());
    Gate1->open();
    Q1.wait();
    CheckFirstDone();
  }
  Q3.wait();

  const ur_event_handle_t SecondHandle = Second->getHandle();
  ASSERT_NE(SecondHandle, nullptr);
  EXPECT_NE(SecondHandle, First->getHandle());
  EXPECT_EQ(handleOf(E), SecondHandle);
  EXPECT_TRUE(isComplete(E));
  // The producer and the consumer launch in the order their gates opened.
  const size_t ProducerLaunch = OlderFirst ? 0 : 1;
  std::lock_guard<std::mutex> Lock(BackendMutex);
  ASSERT_EQ(KernelEvents.size(), 2u);
  EXPECT_EQ(First->getHandle(), KernelEvents[ProducerLaunch]);
  ASSERT_EQ(KernelLaunchWaitLists.size(), 2u);
  EXPECT_EQ(KernelLaunchWaitLists[1 - ProducerLaunch],
            std::vector<ur_event_handle_t>{SecondHandle});
}

INSTANTIATE_TEST_SUITE_P(ReusableEventsLifetime, ReusableEventsCleanupOrderTest,
                         ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool> &Info) {
                           return Info.param ? "OlderFirst" : "NewerFirst";
                         });

// tests-10-02 U28. An event goes through an immediate, a deferred and another
// immediate signal on different queues. Each signal reusing the binding starts
// from a clean state, and every backend event is created with the event's
// properties. Sync point and command-buffer command are seeded directly: the
// graph paths setting them are not exercised here.
TEST_F(ReusableEventsLifetimeTest, ReuseResetsSignalStateKeepsEventProperties) {
  // Out-of-order queues keep no bookkeeping of their last signal, so nothing
  // but the event refers to a signal once it is done.
  sycl::queue Q1{Ctx, Dev, sycl::property::queue::enable_profiling{}};
  sycl::queue Q2{Ctx, Dev};
  sycl::queue Q3{Ctx, Dev};
  sycl::event E = syclex::make_event(
      Ctx,
      syclex::properties{syclex::event_mode{syclex::event_mode_enum::low_power},
                         syclex::enable_profiling{true}});
  auto Impl = sycl::detail::getSyclObjImpl(E);
  auto SoleOwner = [&] {
    return eventually([&] { return Impl->getBinding().use_count() == 1; });
  };
  // Dependencies left over from a previous signal.
  auto Stale = sycl::detail::getSyclObjImpl(syclex::make_event(Ctx));
  auto Seed = [&] {
    sycl::detail::event_binding &B = *Impl->getBinding();
    B.setSyncPoint(7);
    B.setCommandBufferCommand(
        reinterpret_cast<ur_exp_command_buffer_command_handle_t>(0x77));
    std::lock_guard<std::mutex> Lock(B.MMutex);
    B.MPreparedDepsEvents.push_back({Stale->getBinding(), Stale});
    B.MPreparedHostDepsEvents.push_back({Stale->getBinding(), Stale});
  };
  // A deferred signal prepares its own dependencies again: HostDep, if given.
  auto CheckReset = [&](sycl::detail::event_binding &B,
                        const sycl::event *HostDep = nullptr) {
    EXPECT_FALSE(B.MIsFlushed);
    EXPECT_EQ(B.MSyncPoint, 0u);
    EXPECT_EQ(B.MCommandBufferCommand, nullptr);
    EXPECT_EQ(B.MHostProfilingInfo, nullptr);
    std::lock_guard<std::mutex> Lock(B.MMutex);
    EXPECT_TRUE(B.MPreparedDepsEvents.empty());
    if (!HostDep) {
      EXPECT_TRUE(B.MPreparedHostDepsEvents.empty());
      return;
    }
    ASSERT_EQ(B.MPreparedHostDepsEvents.size(), 1u);
    EXPECT_EQ(B.MPreparedHostDepsEvents[0].Event,
              sycl::detail::getSyclObjImpl(*HostDep));
  };

  CompleteAllAtScopeExit CompleteAll;
  setBarriersStayPending(true);
  syclex::enqueue_signal_event(Q1, E);
  setBarriersStayPending(false);
  sycl::detail::event_binding *Binding = Impl->getBinding().get();
  // A consumer on another queue flushes the signal's queue while the signal
  // is pending. Only scheduler commands flush; the buffer takes the consumer
  // there. The consumer and its queue are gone before the binding is reused.
  {
    sycl::queue Consumers = inOrderQueue();
    sycl::buffer<int, 1> Buf{1};
    Consumers.submit([&](sycl::handler &CGH) {
      sycl::accessor A{Buf, CGH, sycl::write_only};
      CGH.depends_on(E);
      CGH.single_task<BindingTestKernel>([]() {});
    });
    EXPECT_TRUE(Binding->MIsFlushed);
    complete(Binding->getHandle());
    Consumers.wait();
  }
  Q1.wait();
  Seed();

  // Deferred, reusing the binding.
  {
    auto Gate = std::make_shared<HostTaskGate>();
    OpenAtScopeExit OpenGate{Gate};
    const sycl::event HostTask = blockQueue(Q2, Gate);
    ASSERT_TRUE(SoleOwner());
    syclex::enqueue_signal_event(Q2, E);
    ASSERT_EQ(Impl->getBinding().get(), Binding)
        << "the deferred signal did not reuse the binding";
    EXPECT_FALSE(Binding->MIsEnqueued);
    EXPECT_EQ(Binding->getHandle(), nullptr);
    CheckReset(*Binding, &HostTask);
    Gate->open();
    Q2.wait();
  }
  EXPECT_TRUE(Binding->MIsEnqueued);
  EXPECT_EQ(Binding->MWorkerQueue.lock(), sycl::detail::getSyclObjImpl(Q2));
  Seed();

  // Immediate again, reusing the binding.
  ASSERT_TRUE(SoleOwner());
  syclex::enqueue_signal_event(Q3, E);
  ASSERT_EQ(Impl->getBinding().get(), Binding)
      << "the immediate signal did not reuse the binding";
  CheckReset(*Binding);
  EXPECT_EQ(Binding->MWorkerQueue.lock(), sycl::detail::getSyclObjImpl(Q3));
  EXPECT_FALSE(Binding->MPotentiallyNativeRecorded);
  Q3.wait();

  // The first backend event, and the one created when the deferred signal
  // reached the backend, have the event's properties.
  std::lock_guard<std::mutex> Lock(BackendMutex);
  ASSERT_EQ(CreatedEvents.size(), 2u);
  for (size_t I = 0; I < CreatedEvents.size(); ++I) {
    EXPECT_TRUE(CreatedEventFlags[I] & UR_EXP_EVENT_FLAG_ENABLE_PROFILING);
    EXPECT_FALSE(CreatedEventFlags[I] & UR_EXP_EVENT_FLAG_IPC_EXP);
    EXPECT_TRUE(CreatedEventLowPower[I]);
  }
}

// tests-10-02 U39. Repeats a deferred signal, a consumer capturing it and a
// wait for the signal's binding. The waiter starts before the backend event is
// published, while it is published, or after it is; it wakes for its own
// signal's completion only.
TEST_F(ReusableEventsLifetimeTest, SignalWaitersWakeForTheirOwnSignal) {
  constexpr int Iterations = 30;
  std::vector<std::weak_ptr<sycl::detail::event_binding>> Bindings;
  std::vector<ur_event_handle_t> Handles;
  std::atomic<int> ConsumersRan{0};
  {
    sycl::queue SignalQueue = inOrderQueue();
    sycl::queue Consumers = inOrderQueue();
    sycl::event E = syclex::make_event(Ctx);
    CompleteAllAtScopeExit CompleteAll;
    setBarriersStayPending(true);
    for (int I = 0; I < Iterations; ++I) {
      auto Gate = std::make_shared<HostTaskGate>();
      OpenAtScopeExit OpenGate{Gate};
      blockQueue(SignalQueue, Gate);
      syclex::enqueue_signal_event(SignalQueue, E);
      std::shared_ptr<sycl::detail::event_binding> Binding = bindingOf(E);
      Consumers.submit([&](sycl::handler &CGH) {
        CGH.depends_on(E);
        CGH.host_task([&] { ++ConsumersRan; });
      });

      std::future<void> Waiter;
      auto StartWaiter = [&] {
        Waiter = std::async(std::launch::async, [Binding] { Binding->wait(); });
      };
      switch (I % 3) {
      case 0:
        StartWaiter();
        // Best effort: the waiter cannot be observed entering the wait.
        EXPECT_TRUE(stillBlocked(Waiter));
        Gate->open();
        break;
      case 1:
        Gate->open();
        StartWaiter();
        break;
      case 2:
        Gate->open();
        ASSERT_TRUE(
            eventually([&] { return Binding->getHandle() != nullptr; }));
        StartWaiter();
        break;
      }
      ASSERT_TRUE(eventually([&] { return Binding->getHandle() != nullptr; }));
      const ur_event_handle_t Handle = Binding->getHandle();
      // Published, but not complete.
      EXPECT_TRUE(stillBlocked(Waiter));
      complete(Handle);
      ASSERT_TRUE(finishes(Waiter));
      EXPECT_TRUE(waitedFor(Handle));
      Bindings.push_back(Binding);
      Handles.push_back(Handle);
    }
    setBarriersStayPending(false);
    SignalQueue.wait();
    Consumers.wait();
    EXPECT_EQ(ConsumersRan, Iterations);
  }
  const std::set<ur_event_handle_t> Distinct(Handles.begin(), Handles.end());
  EXPECT_EQ(Distinct.size(), Handles.size());
  EXPECT_TRUE(eventually([&] {
    return std::all_of(Bindings.begin(), Bindings.end(),
                       [](const auto &B) { return B.expired(); }) &&
           std::all_of(Handles.begin(), Handles.end(), ownershipBalancedLocked);
  }));
}

// tests-10-02 U41. One thread signals an event over and over, while others
// capture it in dependencies, query its status and read its wait list through
// copies of their own. Every captured dependency is a whole signal: the
// backend only ever sees handles the mock created, never a null one.
// Known defect: review-10-01 #2 (MBinding is replaced without
// synchronization). At 8337da70 the test crashes within a few repetitions
// without ThreadSanitizer: SIGSEGV in registerEventDependency, under
// handler::depends_on on the capturing thread.
void signalWhileOthersRead(sycl::queue &SignalQueue, sycl::queue &Consumers,
                           sycl::event E, bool TwoSignalers) {
  constexpr int Rounds = 200;
  std::promise<void> Start;
  std::shared_future<void> Started = Start.get_future().share();
  std::vector<std::thread> Threads;
  auto Signaler = [&, E]() mutable {
    Started.wait();
    for (int I = 0; I < Rounds; ++I)
      syclex::enqueue_signal_event(SignalQueue, E);
  };
  Threads.emplace_back(Signaler);
  if (TwoSignalers)
    Threads.emplace_back(Signaler);
  Threads.emplace_back([&, E] {
    Started.wait();
    for (int I = 0; I < Rounds; ++I)
      Consumers.submit([&](sycl::handler &CGH) {
        CGH.depends_on(E);
        CGH.single_task<BindingTestKernel>([]() {});
      });
  });
  Threads.emplace_back([&, E]() mutable {
    Started.wait();
    for (int I = 0; I < Rounds; ++I) {
      auto Status = E.get_info<sycl::info::event::command_execution_status>();
      EXPECT_TRUE(Status == sycl::info::event_command_status::complete ||
                  Status == sycl::info::event_command_status::submitted ||
                  Status == sycl::info::event_command_status::running);
      for (const sycl::event &Dep : E.get_wait_list())
        EXPECT_NE(sycl::detail::getSyclObjImpl(Dep), nullptr);
    }
  });
  Start.set_value();
  for (std::thread &T : Threads)
    T.join();
  SignalQueue.wait();
  Consumers.wait();

  std::lock_guard<std::mutex> Lock(BackendMutex);
  EXPECT_EQ(KernelLaunchWaitLists.size(), size_t(Rounds));
  for (const auto &WaitList : KernelLaunchWaitLists)
    for (ur_event_handle_t Handle : WaitList)
      EXPECT_TRUE(KnownHandles.count(Handle)) << "a torn or foreign handle";
}

TEST_F(ReusableEventsLifetimeTest, DISABLED_CaptureDuringReassociation) {
  sycl::queue SignalQueue = inOrderQueue();
  sycl::queue Consumers = inOrderQueue();
  sycl::event E = syclex::make_event(Ctx);
  syclex::enqueue_signal_event(SignalQueue, E);
  signalWhileOthersRead(SignalQueue, Consumers, E, /*TwoSignalers*/ false);
}

// Investigation (tests-10-02 U41): two simultaneous signal callers. The
// extension does not say whether they are serialized; only safety is checked.
// Pins behaviour at 8337da70; oracle not agreed. Disabled with the above, as
// the callers race on MBinding the same way (review-10-01 #2); at 8337da70 it
// crashes in most runs, not all.
TEST_F(ReusableEventsLifetimeTest, DISABLED_SimultaneousSignalCallers) {
  sycl::queue SignalQueue = inOrderQueue();
  sycl::queue Consumers = inOrderQueue();
  sycl::event E = syclex::make_event(Ctx);
  signalWhileOthersRead(SignalQueue, Consumers, E, /*TwoSignalers*/ true);
}

// tests-10-02 U42. An executable graph with independent host-task and kernel
// partitions returns the kernel partition's event, which completes only with
// the host task partition (an attachment). The event is then signaled on
// another queue: the new signal does not wait for that attachment, and a
// consumer of the graph's signal still does.
// Known defect: review-10-01 #4 (the attachments are per event, not per
// signal, so they are waited for by every later signal) and review-10-02 #4
// (waits for a captured signal ignore its attachments).
TEST_F(ReusableEventsLifetimeTest,
       DISABLED_CompletionAttachmentsBelongToTheirSignal) {
  sycl::queue GraphQueue{Ctx, Dev};
  sycl::queue SignalQueue = inOrderQueue();
  sycl::queue Consumers = inOrderQueue();
  // Joined after the gate is open: a waiter may be blocked on the host task.
  std::vector<std::future<void>> Waiters;
  auto Gate = std::make_shared<HostTaskGate>();
  OpenAtScopeExit OpenGate{Gate};

  syclex::command_graph<syclex::graph_state::modifiable> Graph{Ctx, Dev};
  // Partitions are numbered from the host tasks: the kernel, following a
  // host task added after the gated one, is in the last partition. The gated
  // host task's partition, without successors, is attached to it.
  Graph.add(
      [&](sycl::handler &CGH) { CGH.host_task([Gate] { Gate->wait(); }); });
  auto Before = Graph.add([&](sycl::handler &CGH) { CGH.host_task([] {}); });
  Graph.add(
      [&](sycl::handler &CGH) { CGH.single_task<BindingTestKernel>([]() {}); },
      {syclex::property::node::depends_on(Before)});
  auto Exec = Graph.finalize();

  sycl::event E = GraphQueue.ext_oneapi_graph(Exec);
  auto Impl = sycl::detail::getSyclObjImpl(E);
  ASSERT_FALSE(Impl->isHost()) << "the graph returned the host task's event";
  ASSERT_EQ(Impl->getPostCompleteEvents().size(), 1u);
  ASSERT_TRUE(Gate->waitEntered());

  std::atomic<bool> ConsumerRan{false};
  Consumers.submit([&](sycl::handler &CGH) {
    CGH.depends_on(E);
    CGH.host_task([&] { ConsumerRan = true; });
  });

  for (int Generation = 0; Generation < 3; ++Generation) {
    syclex::enqueue_signal_event(SignalQueue, E);
    SignalQueue.wait();
    EXPECT_TRUE(Impl->getPostCompleteEvents().empty())
        << "generation " << Generation << " inherited an attachment";
    // Started after the signal is enqueued.
    Waiters.push_back(
        std::async(std::launch::async, [E]() mutable { E.wait(); }));
    EXPECT_EQ(Waiters.back().wait_for(std::chrono::seconds(1)),
              std::future_status::ready)
        << "generation " << Generation
        << ": the new signal waits for the graph's host task";
  }
  // The graph's signal includes its host task partition.
  EXPECT_FALSE(ConsumerRan) << "the consumer did not wait for the host task";

  Gate->open();
  GraphQueue.wait();
  Consumers.wait();
  EXPECT_TRUE(ConsumerRan);
}

} // anonymous namespace
