//==------------ HostTaskAndBarrier.cpp --- Scheduler unit tests------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

#include <helpers/PiMock.hpp>
#include <helpers/ScopedEnvVar.hpp>
#include <helpers/TestKernel.hpp>

#include <detail/global_handler.hpp>

#include <vector>

namespace {
using namespace sycl;
using EventImplPtr = std::shared_ptr<sycl::detail::event_impl>;
using ContextImplPtr = std::shared_ptr<sycl::detail::context_impl>;
using DeviceImplPtr = std::shared_ptr<sycl::detail::device_impl>;

constexpr auto DisableCleanupName = "SYCL_DISABLE_EXECUTION_GRAPH_CLEANUP";

class TestQueueImpl : public sycl::detail::queue_impl {
public:
  TestQueueImpl(ContextImplPtr SyclContext, DeviceImplPtr Dev)
      : sycl::detail::queue_impl(Dev, SyclContext,
                                 SyclContext->get_async_handler(), {}) {}
  using sycl::detail::queue_impl::MDefaultGraphDeps;
  using sycl::detail::queue_impl::MExtGraphDeps;
  using sycl::detail::queue_impl::MMutex;
};

enum TestCGType { KERNEL_TASK, HOST_TASK, BARRIER };

class BarrierHandlingWithHostTask : public ::testing::Test {
protected:
  void SetUp() {
    sycl::platform Plt = Mock.getPlatform();

    sycl::context SyclContext(Plt);
    sycl::device SyclDev =
        sycl::detail::select_device(sycl::default_selector_v, SyclContext);
    QueueDevImpl.reset(
        new TestQueueImpl(sycl::detail::getSyclObjImpl(SyclContext),
                          sycl::detail::getSyclObjImpl(SyclDev)));

    MainLock.lock();
  }

  void TearDown() {
    if (MainLock.owns_lock())
      MainLock.unlock();
  }

  sycl::event AddTask(TestCGType Type, bool BlockHostTask = true) {
    if (Type == TestCGType::HOST_TASK) {
      return QueueDevImpl->submit(
          [&](handler &CGH) {
            CGH.host_task(BlockHostTask ? CustomHostLambda : [] {});
          },
          QueueDevImpl, nullptr, {});
    } else if (Type == TestCGType::KERNEL_TASK) {
      return QueueDevImpl->submit(
          [&](handler &CGH) { CGH.single_task<TestKernel<>>([] {}); },
          QueueDevImpl, nullptr, {});
    } else // (Type == TestCGType::BARRIER)
    {
      return QueueDevImpl->submit(
          [&](handler &CGH) { CGH.ext_oneapi_barrier(); }, QueueDevImpl,
          nullptr, {});
    }
  }

  sycl::event
  InsertBarrierWithWaitList(const std::vector<sycl::event> &WaitList) {
    return QueueDevImpl->submit(
        [&](handler &CGH) { CGH.ext_oneapi_barrier(WaitList); }, QueueDevImpl,
        nullptr, {});
  }

  void BuildAndCheckInnerQueueState(std::vector<EventImplPtr> &Events) {
    {
      std::lock_guard<std::mutex> Guard(QueueDevImpl->MMutex);
      EXPECT_EQ(QueueDevImpl->MDefaultGraphDeps.LastBarrier, nullptr);
      EXPECT_EQ(QueueDevImpl->MDefaultGraphDeps.UnenqueuedCmdEvents.size(), 0u);
    }

    sycl::event BlockedHostTask = AddTask(TestCGType::HOST_TASK);
    EventImplPtr BlockedHostTaskImpl =
        sycl::detail::getSyclObjImpl(BlockedHostTask);
    Events.push_back(BlockedHostTaskImpl);
    {
      std::lock_guard<std::mutex> Guard(QueueDevImpl->MMutex);
      EXPECT_EQ(QueueDevImpl->MDefaultGraphDeps.LastBarrier, nullptr);
      ASSERT_EQ(QueueDevImpl->MDefaultGraphDeps.UnenqueuedCmdEvents.size(), 1u);
      EXPECT_EQ(QueueDevImpl->MDefaultGraphDeps.UnenqueuedCmdEvents[0],
                BlockedHostTaskImpl);
    }

    sycl::event BarrierEvent = AddTask(TestCGType::BARRIER);
    EventImplPtr BarrierEventImpl = sycl::detail::getSyclObjImpl(BarrierEvent);
    {
      std::lock_guard<std::mutex> Guard(QueueDevImpl->MMutex);
      EXPECT_EQ(QueueDevImpl->MDefaultGraphDeps.LastBarrier, BarrierEventImpl);
      EXPECT_EQ(QueueDevImpl->MDefaultGraphDeps.UnenqueuedCmdEvents.size(), 0u);
    }
    Events.push_back(BarrierEventImpl);

    sycl::event KernelEvent = AddTask(TestCGType::KERNEL_TASK);
    EventImplPtr KernelEventImpl = sycl::detail::getSyclObjImpl(KernelEvent);
    {
      std::lock_guard<std::mutex> Guard(QueueDevImpl->MMutex);
      EXPECT_EQ(QueueDevImpl->MDefaultGraphDeps.LastBarrier, BarrierEventImpl);
      ASSERT_EQ(QueueDevImpl->MDefaultGraphDeps.UnenqueuedCmdEvents.size(), 1u);
      EXPECT_EQ(QueueDevImpl->MDefaultGraphDeps.UnenqueuedCmdEvents[0],
                KernelEventImpl);
    }
    Events.push_back(KernelEventImpl);
  }

  sycl::unittest::PiMock Mock;
  sycl::unittest::ScopedEnvVar DisabledCleanup{
      DisableCleanupName, "1",
      sycl::detail::SYCLConfig<
          detail::SYCL_DISABLE_EXECUTION_GRAPH_CLEANUP>::reset};
  std::shared_ptr<TestQueueImpl> QueueDevImpl;

  std::mutex m;
  std::unique_lock<std::mutex> MainLock{m, std::defer_lock};
  std::function<void()> CustomHostLambda = [&]() {
    std::unique_lock<std::mutex> InsideHostTaskLock(this->m);
  };
};

TEST_F(BarrierHandlingWithHostTask, HostTaskBarrierKernel) {
  sycl::event HTEvent = AddTask(TestCGType::HOST_TASK);
  EventImplPtr HostTaskEventImpl = sycl::detail::getSyclObjImpl(HTEvent);
  auto HostTaskWaitList = HostTaskEventImpl->getWaitList();
  EXPECT_EQ(HostTaskWaitList.size(), 0u);
  EXPECT_EQ(HostTaskEventImpl->isEnqueued(), true);

  sycl::event BarrierEvent = AddTask(TestCGType::BARRIER);
  EventImplPtr BarrierEventImpl = sycl::detail::getSyclObjImpl(BarrierEvent);
  auto BarrierWaitList = BarrierEventImpl->getWaitList();
  ASSERT_EQ(BarrierWaitList.size(), 1u);
  EXPECT_EQ(BarrierWaitList[0], HostTaskEventImpl);
  EXPECT_EQ(BarrierEventImpl->isEnqueued(), false);

  sycl::event KernelEvent = AddTask(TestCGType::KERNEL_TASK);
  EventImplPtr KernelEventImpl = sycl::detail::getSyclObjImpl(KernelEvent);
  auto KernelWaitList = KernelEventImpl->getWaitList();
  ASSERT_EQ(KernelWaitList.size(), 1u);
  EXPECT_EQ(KernelWaitList[0], BarrierEventImpl);
  EXPECT_EQ(KernelEventImpl->isEnqueued(), false);

  MainLock.unlock();
  QueueDevImpl->wait();
}

TEST_F(BarrierHandlingWithHostTask, HostTaskKernelBarrier) {
  sycl::event HTEvent = AddTask(TestCGType::HOST_TASK);
  EventImplPtr HostTaskEventImpl = sycl::detail::getSyclObjImpl(HTEvent);
  auto HostTaskWaitList = HostTaskEventImpl->getWaitList();
  EXPECT_EQ(HostTaskWaitList.size(), 0u);
  EXPECT_EQ(HostTaskEventImpl->isEnqueued(), true);

  sycl::event KernelEvent = AddTask(TestCGType::KERNEL_TASK);
  EventImplPtr KernelEventImpl = sycl::detail::getSyclObjImpl(KernelEvent);
  auto KernelWaitList = KernelEventImpl->getWaitList();
  ASSERT_EQ(KernelWaitList.size(), 0u);
  EXPECT_EQ(KernelEventImpl->isEnqueued(), true);

  sycl::event BarrierEvent = AddTask(TestCGType::BARRIER);
  EventImplPtr BarrierEventImpl = sycl::detail::getSyclObjImpl(BarrierEvent);
  auto BarrierWaitList = BarrierEventImpl->getWaitList();
  ASSERT_EQ(BarrierWaitList.size(), 1u);
  EXPECT_EQ(BarrierWaitList[0], HostTaskEventImpl);
  EXPECT_EQ(BarrierEventImpl->isEnqueued(), false);

  MainLock.unlock();
  QueueDevImpl->wait();
}

TEST_F(BarrierHandlingWithHostTask, BarrierHostTaskKernel) {
  sycl::event BarrierEvent = AddTask(TestCGType::BARRIER);
  EventImplPtr BarrierEventImpl = sycl::detail::getSyclObjImpl(BarrierEvent);
  auto BarrierWaitList = BarrierEventImpl->getWaitList();
  ASSERT_EQ(BarrierWaitList.size(), 0u);
  EXPECT_EQ(BarrierEventImpl->isEnqueued(), true);

  sycl::event HTEvent = AddTask(TestCGType::HOST_TASK);
  EventImplPtr HostTaskEventImpl = sycl::detail::getSyclObjImpl(HTEvent);
  auto HostTaskWaitList = HostTaskEventImpl->getWaitList();
  ASSERT_EQ(HostTaskWaitList.size(), 0u);
  EXPECT_EQ(HostTaskEventImpl->isEnqueued(), true);

  sycl::event KernelEvent = AddTask(TestCGType::KERNEL_TASK);
  EventImplPtr KernelEventImpl = sycl::detail::getSyclObjImpl(KernelEvent);
  auto KernelWaitList = KernelEventImpl->getWaitList();
  ASSERT_EQ(KernelWaitList.size(), 0u);
  EXPECT_EQ(KernelEventImpl->isEnqueued(), true);

  MainLock.unlock();
  QueueDevImpl->wait();
}

TEST_F(BarrierHandlingWithHostTask, BarrierKernelHostTask) {
  sycl::event BarrierEvent = AddTask(TestCGType::BARRIER);
  EventImplPtr BarrierEventImpl = sycl::detail::getSyclObjImpl(BarrierEvent);
  auto BarrierWaitList = BarrierEventImpl->getWaitList();
  ASSERT_EQ(BarrierWaitList.size(), 0u);
  EXPECT_EQ(BarrierEventImpl->isEnqueued(), true);

  sycl::event KernelEvent = AddTask(TestCGType::KERNEL_TASK);
  EventImplPtr KernelEventImpl = sycl::detail::getSyclObjImpl(KernelEvent);
  auto KernelWaitList = KernelEventImpl->getWaitList();
  ASSERT_EQ(KernelWaitList.size(), 0u);
  EXPECT_EQ(KernelEventImpl->isEnqueued(), true);

  sycl::event HTEvent = AddTask(TestCGType::HOST_TASK);
  EventImplPtr HostTaskEventImpl = sycl::detail::getSyclObjImpl(HTEvent);
  auto HostTaskWaitList = HostTaskEventImpl->getWaitList();
  ASSERT_EQ(HostTaskWaitList.size(), 0u);
  EXPECT_EQ(HostTaskEventImpl->isEnqueued(), true);

  MainLock.unlock();
  QueueDevImpl->wait();
}

TEST_F(BarrierHandlingWithHostTask, KernelHostTaskBarrier) {
  sycl::event KernelEvent = AddTask(TestCGType::KERNEL_TASK);
  EventImplPtr KernelEventImpl = sycl::detail::getSyclObjImpl(KernelEvent);
  auto KernelWaitList = KernelEventImpl->getWaitList();
  ASSERT_EQ(KernelWaitList.size(), 0u);
  EXPECT_EQ(KernelEventImpl->isEnqueued(), true);

  sycl::event HTEvent = AddTask(TestCGType::HOST_TASK);
  EventImplPtr HostTaskEventImpl = sycl::detail::getSyclObjImpl(HTEvent);
  auto HostTaskWaitList = HostTaskEventImpl->getWaitList();
  ASSERT_EQ(HostTaskWaitList.size(), 0u);
  EXPECT_EQ(HostTaskEventImpl->isEnqueued(), true);

  sycl::event BarrierEvent = AddTask(TestCGType::BARRIER);
  EventImplPtr BarrierEventImpl = sycl::detail::getSyclObjImpl(BarrierEvent);
  auto BarrierWaitList = BarrierEventImpl->getWaitList();
  ASSERT_EQ(BarrierWaitList.size(), 1u);
  EXPECT_EQ(BarrierWaitList[0], HostTaskEventImpl);
  EXPECT_EQ(BarrierEventImpl->isEnqueued(), false);

  MainLock.unlock();
  QueueDevImpl->wait();
}

TEST_F(BarrierHandlingWithHostTask, KernelBarrierHostTask) {
  sycl::event KernelEvent = AddTask(TestCGType::KERNEL_TASK);
  EventImplPtr KernelEventImpl = sycl::detail::getSyclObjImpl(KernelEvent);
  auto KernelWaitList = KernelEventImpl->getWaitList();
  ASSERT_EQ(KernelWaitList.size(), 0u);
  EXPECT_EQ(KernelEventImpl->isEnqueued(), true);

  sycl::event BarrierEvent = AddTask(TestCGType::BARRIER);
  EventImplPtr BarrierEventImpl = sycl::detail::getSyclObjImpl(BarrierEvent);
  auto BarrierWaitList = BarrierEventImpl->getWaitList();
  ASSERT_EQ(BarrierWaitList.size(), 0u);
  EXPECT_EQ(BarrierEventImpl->isEnqueued(), true);

  sycl::event HTEvent = AddTask(TestCGType::HOST_TASK);
  EventImplPtr HostTaskEventImpl = sycl::detail::getSyclObjImpl(HTEvent);
  auto HostTaskWaitList = HostTaskEventImpl->getWaitList();
  ASSERT_EQ(HostTaskWaitList.size(), 0u);
  EXPECT_EQ(HostTaskEventImpl->isEnqueued(), true);

  MainLock.unlock();
  QueueDevImpl->wait();
}

TEST_F(BarrierHandlingWithHostTask, HostTaskUnblockedWaitListBarrierKernel) {
  sycl::event HTEvent = AddTask(TestCGType::HOST_TASK, false);
  EventImplPtr HostTaskEventImpl = sycl::detail::getSyclObjImpl(HTEvent);
  auto HostTaskWaitList = HostTaskEventImpl->getWaitList();
  EXPECT_EQ(HostTaskWaitList.size(), 0u);
  EXPECT_EQ(HostTaskEventImpl->isEnqueued(), true);

  sycl::event BlockedHostTask = AddTask(TestCGType::HOST_TASK);
  EventImplPtr BlockedHostTaskImpl =
      sycl::detail::getSyclObjImpl(BlockedHostTask);
  auto BlockedHostTaskWaitList = BlockedHostTaskImpl->getWaitList();
  EXPECT_EQ(BlockedHostTaskWaitList.size(), 0u);
  EXPECT_EQ(BlockedHostTaskImpl->isEnqueued(), true);

  HostTaskEventImpl->wait(HostTaskEventImpl);

  std::vector<sycl::event> WaitList{HTEvent};
  sycl::event BarrierEvent = InsertBarrierWithWaitList(WaitList);
  EventImplPtr BarrierEventImpl = sycl::detail::getSyclObjImpl(BarrierEvent);
  auto BarrierWaitList = BarrierEventImpl->getWaitList();
  // Events to wait by barrier are stored in a separated vector. Here we are
  // interested in implicit deps only.
  // Host task in barrier wait list could not be handled by backend so it is
  // added by RT to dependency list to initiate deps tracking by scheduler.
  ASSERT_EQ(BarrierWaitList.size(), 1u);
  EXPECT_EQ(BarrierEventImpl->isEnqueued(), true);

  sycl::event KernelEvent = AddTask(TestCGType::KERNEL_TASK);
  EventImplPtr KernelEventImpl = sycl::detail::getSyclObjImpl(KernelEvent);
  auto KernelWaitList = KernelEventImpl->getWaitList();
  ASSERT_EQ(KernelWaitList.size(), 0u);
  EXPECT_EQ(KernelEventImpl->isEnqueued(), true);

  MainLock.unlock();
  QueueDevImpl->wait();
}

TEST_F(BarrierHandlingWithHostTask,
       QueueInnerCleanupOnHostTaskCompletionNotBlocked) {
  // Checks that host task immediately cleans queue fields up if queue mutex is
  // not locked.
  std::vector<EventImplPtr> SubmittedCmdEvents;
  BuildAndCheckInnerQueueState(SubmittedCmdEvents);

  MainLock.unlock();
  QueueDevImpl->wait();
  // Make sure that all host task related stuff is done.
  detail::GlobalHandler::instance().drainThreadPool();
  {
    std::lock_guard<std::mutex> Guard(QueueDevImpl->MMutex);
    EXPECT_EQ(QueueDevImpl->MDefaultGraphDeps.LastBarrier, nullptr);
    EXPECT_EQ(QueueDevImpl->MDefaultGraphDeps.UnenqueuedCmdEvents.size(), 0u);
  }
}

TEST_F(BarrierHandlingWithHostTask,
       QueueInnerCleanupOnHostTaskCompletionBlocked) {
  // Checks that host task postpones cleaning of queue fields if queue mutex is
  // locked. Applicable for graph execution (waits for host task in submit call,
  // if thread is busy with other host task trying to cleanup resources - we
  // could get dead lock) and also better utilizes host task thread.
  std::vector<EventImplPtr> SubmittedCmdEvents;
  BuildAndCheckInnerQueueState(SubmittedCmdEvents);

  {
    // Block queue fields update.
    std::lock_guard<std::mutex> Guard(QueueDevImpl->MMutex);
    MainLock.unlock();
    detail::GlobalHandler::instance().drainThreadPool();
  }
  // Queue mutex was locked and host task was not able to do cleanup.
  {
    std::lock_guard<std::mutex> Guard(QueueDevImpl->MMutex);
    EXPECT_EQ(QueueDevImpl->MDefaultGraphDeps.LastBarrier,
              SubmittedCmdEvents[1]);
    ASSERT_EQ(QueueDevImpl->MDefaultGraphDeps.UnenqueuedCmdEvents.size(), 1u);
    EXPECT_EQ(QueueDevImpl->MDefaultGraphDeps.UnenqueuedCmdEvents[0],
              SubmittedCmdEvents[2]);
  }
  // Wait or new submission will do cleanup. Checks wait.
  QueueDevImpl->wait();
  {
    std::lock_guard<std::mutex> Guard(QueueDevImpl->MMutex);
    EXPECT_EQ(QueueDevImpl->MDefaultGraphDeps.LastBarrier, nullptr);
    EXPECT_EQ(QueueDevImpl->MDefaultGraphDeps.UnenqueuedCmdEvents.size(), 0u);
  }
}

TEST_F(BarrierHandlingWithHostTask,
       QueueInnerCleanupOnHostTaskCompletionBlocked2) {
  // Checks that host task postpones cleaning of queue fields if queue mutex is
  // locked. Applicable for graph execution (waits for host task in submit call,
  // if thread is busy with other host task trying to cleanup resources - we
  // could get dead lock) and also better utilizes host task thread.
  std::vector<EventImplPtr> SubmittedCmdEvents;
  BuildAndCheckInnerQueueState(SubmittedCmdEvents);

  {
    // Block queue fields update.
    std::lock_guard<std::mutex> Guard(QueueDevImpl->MMutex);
    MainLock.unlock();
    detail::GlobalHandler::instance().drainThreadPool();
  }
  // Queue mutex was locked and host task was not able to do cleanup.
  {
    std::lock_guard<std::mutex> Guard(QueueDevImpl->MMutex);
    EXPECT_EQ(QueueDevImpl->MDefaultGraphDeps.LastBarrier,
              SubmittedCmdEvents[1]);
    ASSERT_EQ(QueueDevImpl->MDefaultGraphDeps.UnenqueuedCmdEvents.size(), 1u);
    EXPECT_EQ(QueueDevImpl->MDefaultGraphDeps.UnenqueuedCmdEvents[0],
              SubmittedCmdEvents[2]);
  }
  // Wait or new submission will do cleanup. Checks new submission.
  std::ignore = AddTask(TestCGType::KERNEL_TASK);
  {
    std::lock_guard<std::mutex> Guard(QueueDevImpl->MMutex);
    EXPECT_EQ(QueueDevImpl->MDefaultGraphDeps.LastBarrier, nullptr);
    EXPECT_EQ(QueueDevImpl->MDefaultGraphDeps.UnenqueuedCmdEvents.size(), 0u);
  }
}

} // namespace
