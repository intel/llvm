//==------------ EnqueueWithDependsOnDeps.cpp --- Scheduler unit tests------==//
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

#include <vector>

using namespace sycl;
using EventImplPtr = std::shared_ptr<detail::event_impl>;

constexpr auto DisablePostEnqueueCleanupName =
    "SYCL_DISABLE_POST_ENQUEUE_CLEANUP";

std::vector<std::pair<pi_uint32, const pi_event *>> PassedNumEvents;

bool CheckTestExecutionRequirements(const platform &plt) {
  if (plt.is_host()) {
    std::cout << "Not run due to host-only environment\n";
    return false;
  }
  // This test only contains device image for SPIR-V capable devices.
  if (plt.get_backend() != sycl::backend::opencl &&
      plt.get_backend() != sycl::backend::ext_oneapi_level_zero) {
    std::cout << "Only OpenCL and Level Zero are supported for this test\n";
    return false;
  }
  return true;
}

enum TestCGType { KERNEL_TASK = 0x00, HOST_TASK = 0x01 };

class DependsOnTests : public ::testing::Test {
protected:
  void SetUp() {
    platform Plt = Mock.getPlatform();
    if (!CheckTestExecutionRequirements(Plt))
      GTEST_SKIP();

    queue QueueDev(context(Plt), default_selector_v);
    QueueDevImpl = detail::getSyclObjImpl(QueueDev);
  }

  void TearDown() {}

  detail::Command *
  AddTaskCG(TestCGType Type, const std::vector<EventImplPtr> &Events,
            std::function<void()> *CustomHostLambda = nullptr) {
    std::vector<detail::Command *> ToEnqueue;

    // Emulating processing of command group function
    MockHandlerCustomFinalize MockCGH(QueueDevImpl, false);

    for (auto EventImpl : Events)
      MockCGH.depends_on(detail::createSyclObjFromImpl<event>(EventImpl));

    if (Type == TestCGType::HOST_TASK) {
      if (!CustomHostLambda)
        MockCGH.host_task([] {});
      else
        MockCGH.host_task(*CustomHostLambda);
    } else {
      kernel_bundle KernelBundle =
          sycl::get_kernel_bundle<sycl::bundle_state::input>(
              QueueDevImpl->get_context());
      auto ExecBundle = sycl::build(KernelBundle);
      MockCGH.use_kernel_bundle(ExecBundle);
      MockCGH.single_task<TestKernel<>>([] {});
    }

    std::unique_ptr<sycl::detail::CG> CmdGroup = MockCGH.finalize();

    detail::Command *NewCmd = MS.addCG(
        std::move(CmdGroup),
        Type == TestCGType::HOST_TASK ? MS.getDefaultHostQueue() : QueueDevImpl,
        ToEnqueue);
    EXPECT_EQ(ToEnqueue.size(), 0u);
    return NewCmd;
  }

  void EventsWaitVerification() {
    std::vector<EventImplPtr> Events;

    detail::Command *Cmd1 = AddTaskCG(TestCGType::HOST_TASK, Events);
    EventImplPtr Cmd1Event = Cmd1->getEvent();

    // Depends on host task
    Events.push_back(Cmd1Event);
    detail::Command *Cmd2 = AddTaskCG(TestCGType::KERNEL_TASK, Events);
    EventImplPtr Cmd2Event = Cmd2->getEvent();

    // Depends on kernel depending on host task
    Events.clear();
    Events.push_back(Cmd2Event);
    detail::Command *Cmd3 = AddTaskCG(TestCGType::KERNEL_TASK, Events);
    EventImplPtr Cmd3Event = Cmd3->getEvent();

    std::vector<detail::Command *> BlockedCommands{Cmd2, Cmd3};
    VerifyBlockedCommandsEnqueue(Cmd1, BlockedCommands);

    // One piEventsWait call:
    // kernel2 waits for kernel 1 by sending event list to enqueue launch call
    // (depending on queue property). Cmd3Event.wait() waits for kernel2 via
    // piEventsWait.
    ASSERT_EQ(PassedNumEvents.size(), 1u);
    auto [EventCount, EventArr] = PassedNumEvents[0];
    ASSERT_EQ(EventCount, 1u);
    EXPECT_EQ(*EventArr, Cmd3Event->getHandleRef());
  }

  void VerifyBlockedCommandsEnqueue(
      detail::Command *BlockingCommand,
      std::vector<detail::Command *> &BlockedCommands) {
    std::unique_lock<std::mutex> TestLock(m, std::defer_lock);
    TestLock.lock();
    detail::EnqueueResultT Result;
    for (detail::Command *BlockedCmd : BlockedCommands) {
      EXPECT_FALSE(MS.enqueueCommand(BlockedCmd, Result,
                                     detail::BlockingT::NON_BLOCKING));
      EXPECT_EQ(Result.MResult, detail::EnqueueResultT::SyclEnqueueBlocked);
      EXPECT_EQ(Result.MCmd, static_cast<detail::Command *>(BlockingCommand));
      EXPECT_FALSE(BlockedCmd->isSuccessfullyEnqueued());
    }
    EXPECT_TRUE(BlockingCommand->isSuccessfullyEnqueued());

    TestLock.unlock();

    auto BlockingEvent = BlockingCommand->getEvent();
    BlockingEvent->wait(BlockingEvent);
    {
      auto Lock = MS.acquireOriginSchedGraphWriteLock();
      Lock.lock();
      for (detail::Command *BlockedCmd : BlockedCommands) {
        EXPECT_TRUE(BlockedCmd->isSuccessfullyEnqueued());
      }
    }
    for (detail::Command *BlockedCmd : BlockedCommands) {
      auto BlockedEvent = BlockedCmd->getEvent();
      BlockedEvent->wait(BlockedEvent);
    }
  }

  unittest::PiMock Mock;
  unittest::ScopedEnvVar DisabledCleanup{
      DisablePostEnqueueCleanupName, "1",
      detail::SYCLConfig<detail::SYCL_DISABLE_POST_ENQUEUE_CLEANUP>::reset};
  MockScheduler MS;

  detail::QueueImplPtr QueueDevImpl;

  std::mutex m;
  std::function<void()> CustomHostLambda = [&]() {
    std::unique_lock<std::mutex> InsideHostTaskLock(this->m);
  };
};

TEST_F(DependsOnTests, EnqueueNoMemObjTwoHostTasks) {
  // Checks enqueue of two dependent host tasks
  detail::QueueImplPtr QueueHostImpl = MS.getDefaultHostQueue();
  std::vector<EventImplPtr> Events;

  detail::Command *Cmd1 =
      AddTaskCG(TestCGType::HOST_TASK, Events, &CustomHostLambda);
  EventImplPtr Cmd1Event = Cmd1->getEvent();

  // Simulate depends_on() call
  Events.push_back(Cmd1Event);
  detail::Command *Cmd2 = AddTaskCG(TestCGType::HOST_TASK, Events);
  EventImplPtr Cmd2Event = Cmd2->getEvent();

  std::vector<detail::Command *> BlockedCommands{Cmd2};
  VerifyBlockedCommandsEnqueue(Cmd1, BlockedCommands);
  EXPECT_EQ(Cmd1Event->get_info<info::event::command_execution_status>(),
            info::event_command_status::complete);
  EXPECT_EQ(Cmd2Event->get_info<info::event::command_execution_status>(),
            info::event_command_status::complete);
}

TEST_F(DependsOnTests, EnqueueNoMemObjKernelDepHost) {
  // Checks enqueue of kernel depending on host task
  std::vector<EventImplPtr> Events;

  detail::Command *Cmd1 =
      AddTaskCG(TestCGType::HOST_TASK, Events, &CustomHostLambda);
  EventImplPtr Cmd1Event = Cmd1->getEvent();
  EXPECT_TRUE(Cmd1->isBlocking());

  // Simulate depends_on() call
  Events.push_back(Cmd1Event);
  detail::Command *Cmd2 = AddTaskCG(TestCGType::KERNEL_TASK, Events);

  std::vector<detail::Command *> BlockedCommands{Cmd2};
  VerifyBlockedCommandsEnqueue(Cmd1, BlockedCommands);
}

TEST_F(DependsOnTests, EnqueueNoMemObjHostDepKernel) {
  // Checks enqueue of host task depending on kernel
  std::vector<EventImplPtr> Events;

  detail::Command *Cmd1 = AddTaskCG(TestCGType::KERNEL_TASK, Events);
  EventImplPtr Cmd1Event = Cmd1->getEvent();

  // Simulate depends_on() call
  Events.push_back(Cmd1Event);
  detail::Command *Cmd2 = AddTaskCG(TestCGType::HOST_TASK, Events);
  EventImplPtr Cmd2Event = Cmd2->getEvent();

  detail::EnqueueResultT Result;
  EXPECT_TRUE(MS.enqueueCommand(Cmd2, Result, detail::BlockingT::NON_BLOCKING));

  EXPECT_TRUE(Cmd1->isSuccessfullyEnqueued());
  EXPECT_TRUE(Cmd2->isSuccessfullyEnqueued());
  Cmd2Event->wait(Cmd2Event);
}

TEST_F(DependsOnTests, EnqueueNoMemObjDoubleKernelDepHostBlocked) {
  // Checks blocking command tranfer for dependent kernels and enqueue of
  // kernels on host task completion
  std::vector<EventImplPtr> Events;

  detail::Command *Cmd1 = AddTaskCG(TestCGType::HOST_TASK, Events);
  EventImplPtr Cmd1Event = Cmd1->getEvent();
  Cmd1->MIsBlockable = true;
  Cmd1->MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueBlocked;

  // Depends on host task
  Events.push_back(Cmd1Event);
  detail::Command *Cmd2 = AddTaskCG(TestCGType::KERNEL_TASK, Events);
  EventImplPtr Cmd2Event = Cmd2->getEvent();

  // Depends on kernel depending on host task
  Events.clear();
  Events.push_back(Cmd2Event);
  detail::Command *Cmd3 = AddTaskCG(TestCGType::KERNEL_TASK, Events);
  EventImplPtr Cmd3Event = Cmd3->getEvent();

  detail::EnqueueResultT Result;
  EXPECT_FALSE(
      MS.enqueueCommand(Cmd2, Result, detail::BlockingT::NON_BLOCKING));
  EXPECT_EQ(Result.MResult, detail::EnqueueResultT::SyclEnqueueBlocked);
  EXPECT_EQ(Result.MCmd, static_cast<detail::Command *>(Cmd1));
  EXPECT_FALSE(
      MS.enqueueCommand(Cmd3, Result, detail::BlockingT::NON_BLOCKING));
  EXPECT_EQ(Result.MResult, detail::EnqueueResultT::SyclEnqueueBlocked);
  EXPECT_EQ(Result.MCmd, static_cast<detail::Command *>(Cmd1));

  // Preconditions for post enqueue checks
  EXPECT_FALSE(Cmd1->isSuccessfullyEnqueued());
  EXPECT_FALSE(Cmd2->isSuccessfullyEnqueued());
  EXPECT_FALSE(Cmd3->isSuccessfullyEnqueued());

  Cmd1->MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueReady;

  std::vector<detail::Command *> BlockedCommands{Cmd2, Cmd3};
  VerifyBlockedCommandsEnqueue(Cmd1, BlockedCommands);
}

TEST_F(DependsOnTests, EnqueueNoMemObjDoubleKernelDepHost) {
  // Checks blocking command tranfer for dependent kernels and enqueue of
  // kernels on host task completion
  std::vector<EventImplPtr> Events;

  detail::Command *Cmd1 = AddTaskCG(TestCGType::HOST_TASK, Events);
  EventImplPtr Cmd1Event = Cmd1->getEvent();
  Cmd1->MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueBlocked;

  // Depends on host task
  Events.push_back(Cmd1Event);
  detail::Command *Cmd2 = AddTaskCG(TestCGType::KERNEL_TASK, Events);
  EventImplPtr Cmd2Event = Cmd2->getEvent();

  // Depends on kernel depending on host task
  Events.clear();
  Events.push_back(Cmd2Event);
  detail::Command *Cmd3 = AddTaskCG(TestCGType::KERNEL_TASK, Events);
  EventImplPtr Cmd3Event = Cmd3->getEvent();

  std::vector<detail::Command *> BlockedCommands{Cmd2, Cmd3};
  VerifyBlockedCommandsEnqueue(Cmd1, BlockedCommands);
}
