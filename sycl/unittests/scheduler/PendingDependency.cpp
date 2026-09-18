//==-- PendingDependency.cpp --- Detecting dependencies pending in the RT --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Phase 1 of option O7 for full Reusable Events support: the mechanism that
// tells whether an event is a dependency of a command which the SYCL runtime
// still holds. Such a command reads the event again when it is finally
// enqueued, so re-associating the event before that happens would retarget the
// dependency.
//
// The count is taken in Command::processDepEvent, where a dependency is
// recorded, and given back in Command::enqueue once the command has passed its
// dependencies to the backend, or in ~Command if it never got that far.
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

#include <helpers/TestKernel.hpp>
#include <helpers/UrMock.hpp>

#include <detail/event_impl.hpp>

#include <condition_variable>
#include <mutex>

namespace {
using namespace sycl;

// A gate for a host task to block on, so that the queue is left with a command
// the runtime has not passed to the backend.
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

detail::event_impl &imp(const event &E) { return *detail::getSyclObjImpl(E); }

// Set just before the host task is submitted, read on the host task thread.
std::atomic<detail::event_impl *> WatchedEvent{nullptr};
std::atomic<uint32_t> CountDuringDepRead{0};
std::atomic<bool> DepReadObserved{false};

// Runs on the host task thread, from DispatchHostTask::waitForEvents, i.e. at
// the exact point where the host task reads its dependencies. Records once, so
// that later waits do not overwrite the observation.
ur_result_t recordCountOnUrEventWait(void *) {
  if (detail::event_impl *E = WatchedEvent.load())
    if (!DepReadObserved.exchange(true))
      CountDuringDepRead.store(E->getUnenqueuedDependentCount());
  return UR_RESULT_SUCCESS;
}

// Submits a host task that blocks on Gate, so that everything submitted to the
// in-order queue after it stays inside the runtime.
event blockQueue(queue &Q, HostTaskGate &Gate) {
  return Q.submit([&](handler &CGH) { CGH.host_task([&] { Gate.wait(); }); });
}

class PendingDependencyTest : public ::testing::Test {
protected:
  unittest::UrMock<> Mock;
};

// A dependency that goes straight to the backend is never pending: the command
// reads the event during submission, so there is nothing left to hold.
TEST_F(PendingDependencyTest, BypassedCommandHoldsNothing) {
  queue Q{platform().get_devices()[0]};

  event Producer = Q.single_task<TestKernel>([] {});
  Q.wait();
  EXPECT_EQ(imp(Producer).getUnenqueuedDependentCount(), 0u);

  event Consumer = Q.submit([&](handler &CGH) {
    CGH.depends_on(Producer);
    CGH.single_task<TestKernel>([] {});
  });
  Q.wait();

  EXPECT_FALSE(imp(Producer).hasUnenqueuedDependents());
  EXPECT_FALSE(imp(Consumer).hasUnenqueuedDependents());
}

// A command held behind a host task keeps its dependency pending until it
// reaches the backend.
TEST_F(PendingDependencyTest, CommandBehindHostTaskHoldsItsDependency) {
  queue Q{platform().get_devices()[0], property::queue::in_order()};

  event Producer = Q.single_task<TestKernel>([] {});
  Q.wait();
  ASSERT_FALSE(imp(Producer).hasUnenqueuedDependents());

  HostTaskGate Gate;
  event Blocker = blockQueue(Q, Gate);

  Q.submit([&](handler &CGH) {
    CGH.depends_on(Producer);
    CGH.single_task<TestKernel>([] {});
  });

  // The kernel is queued in the runtime, so Producer is a pending dependency.
  EXPECT_TRUE(imp(Producer).hasUnenqueuedDependents());

  Gate.open();
  Q.wait();

  EXPECT_FALSE(imp(Producer).hasUnenqueuedDependents());
}

// Every holding command is counted, and each releases its own count.
TEST_F(PendingDependencyTest, CountsEveryHoldingCommand) {
  queue Q{platform().get_devices()[0], property::queue::in_order()};

  event Producer = Q.single_task<TestKernel>([] {});
  Q.wait();

  HostTaskGate Gate;
  event Blocker = blockQueue(Q, Gate);

  const uint32_t Before = imp(Producer).getUnenqueuedDependentCount();

  for (int I = 0; I < 3; ++I)
    Q.submit([&](handler &CGH) {
      CGH.depends_on(Producer);
      CGH.single_task<TestKernel>([] {});
    });

  EXPECT_EQ(imp(Producer).getUnenqueuedDependentCount(), Before + 3);

  Gate.open();
  Q.wait();

  EXPECT_EQ(imp(Producer).getUnenqueuedDependentCount(), 0u);
}

// A host task is a dependency the runtime resolves itself, so its event is
// pending for the commands queued behind it as well.
TEST_F(PendingDependencyTest, HostTaskEventIsPendingForItsSuccessor) {
  queue Q{platform().get_devices()[0], property::queue::in_order()};

  HostTaskGate Gate;
  event Blocker = blockQueue(Q, Gate);

  Q.submit([&](handler &CGH) { CGH.single_task<TestKernel>([] {}); });

  EXPECT_TRUE(imp(Blocker).hasUnenqueuedDependents());

  Gate.open();
  Q.wait();

  EXPECT_FALSE(imp(Blocker).hasUnenqueuedDependents());
}

// A command blocked by a live host accessor holds its dependency for as long as
// the accessor lives - the case that makes the synchronous path in O7 unsafe.
TEST_F(PendingDependencyTest, CommandBehindHostAccessorHoldsItsDependency) {
  queue Q{platform().get_devices()[0], property::queue::in_order()};

  event Producer = Q.single_task<TestKernel>([] {});
  Q.wait();
  ASSERT_FALSE(imp(Producer).hasUnenqueuedDependents());

  buffer<int, 1> Buf{range<1>{1}};
  {
    auto HostAcc = Buf.get_host_access();

    Q.submit([&](handler &CGH) {
      CGH.depends_on(Producer);
      accessor Acc{Buf, CGH, write_only};
      CGH.fill(Acc, 0);
    });

    EXPECT_TRUE(imp(Producer).hasUnenqueuedDependents());
  }

  Q.wait();

  EXPECT_FALSE(imp(Producer).hasUnenqueuedDependents());
}

// A host task reads its dependencies on the host task thread when it runs, in
// DispatchHostTask::waitForEvents, not when it is enqueued - enqueueing one
// only hands the job to the thread pool. So it has to keep them counted until
// then, which is observed here from inside the read itself.
TEST_F(PendingDependencyTest, HostTaskHoldsItsDependencyUntilItReadsIt) {
  mock::getCallbacks().set_replace_callback("urEventWait",
                                            &recordCountOnUrEventWait);

  queue Q{platform().get_devices()[0]};

  event Producer = Q.single_task<TestKernel>([] {});
  ASSERT_NE(imp(Producer).getHandle(), nullptr)
      << "the host task must have a backend event to wait for";

  CountDuringDepRead.store(0);
  DepReadObserved.store(false);
  WatchedEvent.store(&imp(Producer));

  Q.submit([&](handler &CGH) {
    CGH.depends_on(Producer);
    CGH.host_task([] {});
  });

  Q.wait();
  WatchedEvent.store(nullptr);

  ASSERT_TRUE(DepReadObserved.load())
      << "the host task was expected to wait for its dependency";
  EXPECT_EQ(CountDuringDepRead.load(), 1u);
  EXPECT_FALSE(imp(Producer).hasUnenqueuedDependents());
}

// An event that has not been signaled yet is counted as well, and it has to be.
// Such an event has no backend event, so processDepEvent files it under the
// host dependencies, where the holding command calls waitInternal() on it once
// it is finally enqueued. By then the event may have acquired a backend event -
// which is exactly what enqueue_signal_event does - and the wait binds to that
// later signal. So the dependency does retarget, and the count must see it.
TEST_F(PendingDependencyTest, UnsignaledEventIsCountedAsPending) {
  queue Q{platform().get_devices()[0], property::queue::in_order()};

  event Unsignaled;
  ASSERT_TRUE(imp(Unsignaled).isDefaultConstructed());
  ASSERT_EQ(imp(Unsignaled).getHandle(), nullptr);

  HostTaskGate Gate;
  event Blocker = blockQueue(Q, Gate);

  Q.submit([&](handler &CGH) {
    CGH.depends_on(Unsignaled);
    CGH.single_task<TestKernel>([] {});
  });

  EXPECT_TRUE(imp(Unsignaled).hasUnenqueuedDependents());

  Gate.open();
  Q.wait();

  EXPECT_FALSE(imp(Unsignaled).hasUnenqueuedDependents());
}

// The events of a barrier with a wait list are kept by the command group
// instead of the dependency lists processDepEvent fills, so they are counted
// separately, in ExecCGCommand's constructor. An event which has not been
// signaled yet is counted there as well: the barrier reads the wait list when
// it is enqueued, and by then a signal may have given the event a backend
// event, which the barrier would wait for instead of treating the event as
// nothing to wait for.
TEST_F(PendingDependencyTest, BarrierWaitListIsCountedIncludingUnsignaled) {
  queue Q{platform().get_devices()[0], property::queue::in_order()};

  event Signaled = Q.single_task<TestKernel>([] {});
  Q.wait();
  ASSERT_NE(imp(Signaled).getHandle(), nullptr);

  event Unsignaled;
  ASSERT_TRUE(imp(Unsignaled).isDefaultConstructed());
  ASSERT_EQ(imp(Unsignaled).getHandle(), nullptr);

  HostTaskGate Gate;
  event Blocker = blockQueue(Q, Gate);

  Q.submit([&](handler &CGH) {
    CGH.ext_oneapi_barrier({Signaled, Unsignaled});
  });

  EXPECT_TRUE(imp(Signaled).hasUnenqueuedDependents());
  EXPECT_TRUE(imp(Unsignaled).hasUnenqueuedDependents());

  Gate.open();
  Q.wait();

  EXPECT_FALSE(imp(Signaled).hasUnenqueuedDependents());
  EXPECT_FALSE(imp(Unsignaled).hasUnenqueuedDependents());
}

} // anonymous namespace
