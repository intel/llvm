//==------------ EnqueueWithHostTaskDeps.cpp --- Scheduler unit tests
//----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"
#include <detail/handler_impl.hpp>

#include <helpers/CommonRedefinitions.hpp>
#include <helpers/PiMock.hpp>
#include <helpers/ScopedEnvVar.hpp>
#include <helpers/TestKernel.hpp>

using namespace sycl;
using EventImplPtr = std::shared_ptr<detail::event_impl>;

class MockHandlerEnqueueHostDeps : public MockHandler {
public:
  MockHandlerEnqueueHostDeps(std::shared_ptr<sycl::detail::queue_impl> Queue,
                             bool IsHost)
      : MockHandler(Queue, IsHost) {}

  std::unique_ptr<sycl::detail::CG> finalize() {
    std::shared_ptr<sycl::detail::handler_impl> Impl = evictHandlerImpl();
    std::unique_ptr<sycl::detail::CG> CommandGroup;
    switch (getType()) {
    case sycl::detail::CG::Kernel: {
      CommandGroup.reset(new sycl::detail::CGExecKernel(
          getNDRDesc(), std::move(getHostKernel()), getKernel(),
          getArgsStorage(), getAccStorage(), getSharedPtrStorage(),
          getRequirements(), getEvents(), getArgs(), getKernelName(),
          getOSModuleHandle(), getStreamStorage(), Impl->MAuxiliaryResources,
          getCGType(), getCodeLoc()));
      break;
    }
    case sycl::detail::CG::CodeplayHostTask: {
      CommandGroup.reset(new detail::CGHostTask(
          std::move(getHostTask()), getQueue(), getQueue()->getContextImplPtr(),
          getArgs(), getArgsStorage(), getAccStorage(), getSharedPtrStorage(),
          getRequirements(), getEvents(), getCGType(), getCodeLoc()));
      break;
    }
    default:
      throw sycl::runtime_error("Unhandled type of command group",
                                PI_INVALID_OPERATION);
    }

    return CommandGroup;
  }
};

detail::Command *AddTaskCG(bool IsHost, MockScheduler &MS,
                           detail::QueueImplPtr DevQueue,
                           const std::vector<EventImplPtr> &Events) {
  std::vector<detail::Command *> ToEnqueue;

  kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(
          DevQueue->get_context());
  auto ExecBundle = sycl::build(KernelBundle);

  // Emulating processing of command group function
  MockHandlerEnqueueHostDeps MockCGH(DevQueue, false);
  MockCGH.use_kernel_bundle(ExecBundle);

  for (auto EventImpl : Events)
    MockCGH.depends_on(detail::createSyclObjFromImpl<event>(EventImpl));

  if (IsHost)
    MockCGH.host_task([] {});
  else
    MockCGH.single_task<TestKernel>([] {});

  std::unique_ptr<sycl::detail::CG> CmdGroup = MockCGH.finalize();

  detail::Command *NewCmd =
      MS.addCG(std::move(CmdGroup),
               IsHost ? MS.getDefaultHostQueue() : DevQueue, ToEnqueue);
  EXPECT_EQ(ToEnqueue.size(), 0u);
  return NewCmd;
}

void VerifyNewHostTaskValidness(
    detail::Command *NewCmd, const std::vector<EventImplPtr> &BlockingTasks) {
  ASSERT_NE(NewCmd, nullptr);
  EXPECT_EQ(NewCmd->getType(), detail::Command::RUN_CG);

  EventImplPtr NewCmdEvent = NewCmd->getEvent();
  ASSERT_NE(NewCmdEvent, nullptr);

  EventImplPtr NewCmdEmptyCmdEvent = NewCmdEvent->getEmptyCommandEvent();
  ASSERT_NE(NewCmdEmptyCmdEvent, nullptr);

  detail::Command *NewCmdEmptyCmd =
      static_cast<detail::Command *>(NewCmdEmptyCmdEvent->getCommand());
  ASSERT_NE(NewCmdEmptyCmd, nullptr);
  EXPECT_EQ(NewCmdEmptyCmd->getType(), detail::Command::EMPTY_TASK);

  EXPECT_EQ(NewCmdEmptyCmd,
            (static_cast<detail::ExecCGCommand *>(NewCmd))->MEmptyCmd);
  EXPECT_EQ(NewCmdEvent->getBlockingExplicitDeps().size(),
            BlockingTasks.size());
  if (BlockingTasks.size()) {
    auto &BlockingExplicitDeps = NewCmdEvent->getBlockingExplicitDeps();
    EXPECT_TRUE(std::equal(BlockingExplicitDeps.begin(),
                           BlockingExplicitDeps.end(), BlockingTasks.begin()));

    EXPECT_TRUE(std::all_of(
        BlockingTasks.cbegin(), BlockingTasks.cend(),
        [&NewCmdEvent](EventImplPtr BlockingTask) {
          const auto &BlockedUsers =
              (static_cast<detail::EmptyCommand *>(BlockingTask->getCommand()))
                  ->getBlockedUsers();
          return BlockedUsers.find(NewCmdEvent) != BlockedUsers.end();
        }));
  }
}

void VerifyNewSingleTaskValidness(
    detail::Command *NewCmd, const std::vector<EventImplPtr> &BlockingTasks) {
  ASSERT_NE(NewCmd, nullptr);
  EXPECT_EQ(NewCmd->getType(), detail::Command::RUN_CG);

  EventImplPtr NewCmdEvent = NewCmd->getEvent();
  ASSERT_NE(NewCmdEvent, nullptr);

  EventImplPtr NewCmdEmptyCmdEvent = NewCmdEvent->getEmptyCommandEvent();
  ASSERT_EQ(NewCmdEmptyCmdEvent, nullptr);

  EXPECT_EQ(NewCmdEvent->getBlockingExplicitDeps().size(),
            BlockingTasks.size());
  if (BlockingTasks.size()) {
    auto &BlockingExplicitDeps = NewCmdEvent->getBlockingExplicitDeps();
    EXPECT_TRUE(std::equal(BlockingExplicitDeps.begin(),
                           BlockingExplicitDeps.end(), BlockingTasks.begin()));

    EXPECT_TRUE(std::all_of(
        BlockingTasks.cbegin(), BlockingTasks.cend(),
        [&NewCmdEvent](EventImplPtr BlockingTask) {
          const auto &BlockedUsers =
              (static_cast<detail::EmptyCommand *>(BlockingTask->getCommand()))
                  ->getBlockedUsers();
          return BlockedUsers.find(NewCmdEvent) != BlockedUsers.end();
        }));
  }
}

inline constexpr auto DisablePostEnqueueCleanupName =
    "SYCL_DISABLE_POST_ENQUEUE_CLEANUP";

TEST_F(SchedulerTest, EnqueueNoMemObjTwoHostTasks) {
  // Checks enqueue of two dependent host tasks

  unittest::ScopedEnvVar DisabledCleanup{
      DisablePostEnqueueCleanupName, "1",
      detail::SYCLConfig<detail::SYCL_DISABLE_POST_ENQUEUE_CLEANUP>::reset};

  default_selector Selector;
  platform Plt{Selector};
  if (Plt.is_host()) {
    std::cout << "Not run due to host-only environment\n";
    return;
  }
  queue QueueDev(context(Plt), Selector);
  MockScheduler MS;

  detail::QueueImplPtr QueueDevImpl = detail::getSyclObjImpl(QueueDev);
  detail::QueueImplPtr QueueHostImpl = MS.getDefaultHostQueue();

  std::vector<EventImplPtr> Events;
  std::vector<EventImplPtr> BlockingTaskEvents;

  detail::Command *Cmd1 = AddTaskCG(true, MS, QueueDevImpl, Events);
  EventImplPtr Cmd1Event = Cmd1->getEvent();
  VerifyNewHostTaskValidness(Cmd1, BlockingTaskEvents);

  // Simulate depends_on() call
  Events.push_back(Cmd1Event);
  BlockingTaskEvents.push_back(Cmd1->getEvent()->getEmptyCommandEvent());
  detail::Command *Cmd2 = AddTaskCG(true, MS, QueueDevImpl, Events);
  EventImplPtr Cmd2Event = Cmd2->getEvent();
  VerifyNewHostTaskValidness(Cmd2, BlockingTaskEvents);

  // Not expect any execution here!
  detail::EnqueueResultT Result;
  EXPECT_FALSE(
      MS.enqueueCommand(Cmd2, Result, detail::BlockingT::NON_BLOCKING));
  EXPECT_EQ(Result.MResult, detail::EnqueueResultT::SyclEnqueueBlocked);
  EXPECT_EQ(Result.MCmd,
            (static_cast<detail::ExecCGCommand *>(Cmd1))->MEmptyCmd);

  // Preconditions for post enqueue checks
  EXPECT_FALSE(Cmd1->isSuccessfullyEnqueued());
  EXPECT_FALSE(Cmd2->isSuccessfullyEnqueued());
  EXPECT_EQ(Cmd1Event->get_info<info::event::command_execution_status>(),
            info::event_command_status::submitted);
  EXPECT_EQ(Cmd2Event->get_info<info::event::command_execution_status>(),
            info::event_command_status::submitted);

  EXPECT_TRUE(MS.enqueueCommand(Cmd1, Result, detail::BlockingT::NON_BLOCKING));
  // Wait will cleanup Cmd1 - not able to use any more, but Cmd2 should not be
  // cleaned up yet since we disable post enqueue cleanup.
  Cmd1Event->wait(Cmd1Event);
  EXPECT_EQ(Cmd1Event->get_info<info::event::command_execution_status>(),
            info::event_command_status::complete);
  {
    // Write lock allows to wait till all actions on host task completion are
    // executed including blocked users enqueue
    auto Lock = MS.acquireOriginSchedGraphWriteLock();
    Lock.lock();
    EXPECT_TRUE(Cmd2->isSuccessfullyEnqueued());
  }
  Cmd2Event->wait(Cmd2Event);
}

TEST_F(SchedulerTest, EnqueueNoMemObjKernelDepHost) {
  // Checks enqueue of kernel depending on host task
  unittest::ScopedEnvVar DisabledCleanup{
      DisablePostEnqueueCleanupName, "1",
      detail::SYCLConfig<detail::SYCL_DISABLE_POST_ENQUEUE_CLEANUP>::reset};

  default_selector Selector;
  platform Plt{Selector};
  if (Plt.is_host()) {
    std::cout << "Not run due to host-only environment\n";
    return;
  }
  unittest::PiMock Mock{Plt};
  setupDefaultMockAPIs(Mock);

  queue QueueDev(context(Plt), Selector);
  MockScheduler MS;

  detail::QueueImplPtr QueueDevImpl = detail::getSyclObjImpl(QueueDev);

  std::vector<EventImplPtr> Events;
  std::vector<EventImplPtr> BlockingTaskEvents;

  detail::Command *Cmd1 = AddTaskCG(true, MS, QueueDevImpl, Events);
  EventImplPtr Cmd1Event = Cmd1->getEvent();
  VerifyNewHostTaskValidness(Cmd1, BlockingTaskEvents);

  // Simulate depends_on() call
  Events.push_back(Cmd1Event);
  BlockingTaskEvents.push_back(Cmd1->getEvent()->getEmptyCommandEvent());
  detail::Command *Cmd2 = AddTaskCG(false, MS, QueueDevImpl, Events);
  EventImplPtr Cmd2Event = Cmd2->getEvent();
  VerifyNewSingleTaskValidness(Cmd2, BlockingTaskEvents);

  // Not expect any execution here!
  detail::EnqueueResultT Result;
  EXPECT_FALSE(
      MS.enqueueCommand(Cmd2, Result, detail::BlockingT::NON_BLOCKING));
  EXPECT_EQ(Result.MResult, detail::EnqueueResultT::SyclEnqueueBlocked);
  EXPECT_EQ(Result.MCmd,
            (static_cast<detail::ExecCGCommand *>(Cmd1))->MEmptyCmd);

  // Preconditions for post enqueue checks
  EXPECT_FALSE(Cmd1->isSuccessfullyEnqueued());
  EXPECT_FALSE(Cmd2->isSuccessfullyEnqueued());
  EXPECT_EQ(Cmd1Event->get_info<info::event::command_execution_status>(),
            info::event_command_status::submitted);
  EXPECT_EQ(Cmd2Event->get_info<info::event::command_execution_status>(),
            info::event_command_status::submitted);

  EXPECT_TRUE(MS.enqueueCommand(Cmd1, Result, detail::BlockingT::NON_BLOCKING));
  // Wait will cleanup Cmd1 - not able to use any more, but Cmd2 should not be
  // cleaned up yet since we disable post enqueue cleanup.
  Cmd1Event->wait(Cmd1Event);
  EXPECT_EQ(Cmd1Event->get_info<info::event::command_execution_status>(),
            info::event_command_status::complete);
  {
    // Write lock allows to wait till all actions on host task completion are
    // executed including blocked users enqueue
    auto Lock = MS.acquireOriginSchedGraphWriteLock();
    Lock.lock();
    EXPECT_TRUE(Cmd2->isSuccessfullyEnqueued());
  }
  Cmd2Event->wait(Cmd2Event);
}

TEST_F(SchedulerTest, EnqueueNoMemObjHostDepKernel) {
  // Checks enqueue of host task depending on kernel
  unittest::ScopedEnvVar DisabledCleanup{
      DisablePostEnqueueCleanupName, "1",
      detail::SYCLConfig<detail::SYCL_DISABLE_POST_ENQUEUE_CLEANUP>::reset};

  default_selector Selector;
  platform Plt{Selector};
  if (Plt.is_host()) {
    std::cout << "Not run due to host-only environment\n";
    return;
  }
  unittest::PiMock Mock{Plt};
  setupDefaultMockAPIs(Mock);

  queue QueueDev(context(Plt), Selector);
  MockScheduler MS;

  detail::QueueImplPtr QueueDevImpl = detail::getSyclObjImpl(QueueDev);

  std::vector<EventImplPtr> Events;
  std::vector<EventImplPtr> BlockingTaskEvents;

  detail::Command *Cmd1 = AddTaskCG(false, MS, QueueDevImpl, Events);
  EventImplPtr Cmd1Event = Cmd1->getEvent();
  VerifyNewSingleTaskValidness(Cmd1, BlockingTaskEvents);

  // Simulate depends_on() call
  Events.push_back(Cmd1Event);
  BlockingTaskEvents.clear();
  detail::Command *Cmd2 = AddTaskCG(true, MS, QueueDevImpl, Events);
  EventImplPtr Cmd2Event = Cmd2->getEvent();
  VerifyNewHostTaskValidness(Cmd2, BlockingTaskEvents);

  // Not expect any execution here!
  detail::EnqueueResultT Result;
  EXPECT_TRUE(MS.enqueueCommand(Cmd1, Result, detail::BlockingT::NON_BLOCKING));
  EXPECT_TRUE(MS.enqueueCommand(Cmd2, Result, detail::BlockingT::NON_BLOCKING));

  // Preconditions for post enqueue checks
  EXPECT_TRUE(Cmd1->isSuccessfullyEnqueued());
  EXPECT_TRUE(Cmd2->isSuccessfullyEnqueued());
  Cmd2Event->wait(Cmd2Event);
  EXPECT_EQ(Cmd2Event->get_info<info::event::command_execution_status>(),
            info::event_command_status::complete);
}

TEST_F(SchedulerTest, EnqueueNoMemObjDoubleKernelDepHost) {
  // Checks blocking command tranfer for dependent kernels and enqueue of root
  // kernel on host task completion
  unittest::ScopedEnvVar DisabledCleanup{
      DisablePostEnqueueCleanupName, "1",
      detail::SYCLConfig<detail::SYCL_DISABLE_POST_ENQUEUE_CLEANUP>::reset};

  default_selector Selector;
  platform Plt{Selector};
  if (Plt.is_host()) {
    std::cout << "Not run due to host-only environment\n";
    return;
  }
  unittest::PiMock Mock{Plt};
  setupDefaultMockAPIs(Mock);

  queue QueueDev(context(Plt), Selector);
  MockScheduler MS;

  detail::QueueImplPtr QueueDevImpl = detail::getSyclObjImpl(QueueDev);

  std::vector<EventImplPtr> Events;
  std::vector<EventImplPtr> BlockingTaskEvents;

  detail::Command *Cmd1 = AddTaskCG(true, MS, QueueDevImpl, Events);
  EventImplPtr Cmd1Event = Cmd1->getEvent();
  VerifyNewHostTaskValidness(Cmd1, BlockingTaskEvents);

  // Depends on host task
  Events.push_back(Cmd1Event);
  BlockingTaskEvents.push_back(Cmd1->getEvent()->getEmptyCommandEvent());
  detail::Command *Cmd2 = AddTaskCG(false, MS, QueueDevImpl, Events);
  EventImplPtr Cmd2Event = Cmd2->getEvent();
  VerifyNewSingleTaskValidness(Cmd2, BlockingTaskEvents);

  // Depends on kernel depending on host task
  Events.clear();
  Events.push_back(Cmd2Event);
  // We expect to see host task is blocking task for second layer kernel
  ASSERT_EQ(BlockingTaskEvents.size(), 1u);
  detail::Command *Cmd3 = AddTaskCG(false, MS, QueueDevImpl, Events);
  EventImplPtr Cmd3Event = Cmd2->getEvent();
  VerifyNewSingleTaskValidness(Cmd3, BlockingTaskEvents);

  // Not expect any execution here!
  detail::EnqueueResultT Result;
  EXPECT_FALSE(
      MS.enqueueCommand(Cmd2, Result, detail::BlockingT::NON_BLOCKING));
  EXPECT_EQ(Result.MResult, detail::EnqueueResultT::SyclEnqueueBlocked);
  EXPECT_EQ(Result.MCmd,
            (static_cast<detail::ExecCGCommand *>(Cmd1))->MEmptyCmd);
  EXPECT_FALSE(
      MS.enqueueCommand(Cmd3, Result, detail::BlockingT::NON_BLOCKING));
  EXPECT_EQ(Result.MResult, detail::EnqueueResultT::SyclEnqueueBlocked);
  EXPECT_EQ(Result.MCmd,
            (static_cast<detail::ExecCGCommand *>(Cmd1))->MEmptyCmd);

  // Preconditions for post enqueue checks
  EXPECT_FALSE(Cmd1->isSuccessfullyEnqueued());
  EXPECT_FALSE(Cmd2->isSuccessfullyEnqueued());
  EXPECT_FALSE(Cmd3->isSuccessfullyEnqueued());
  EXPECT_EQ(Cmd1Event->get_info<info::event::command_execution_status>(),
            info::event_command_status::submitted);

  EXPECT_TRUE(MS.enqueueCommand(Cmd1, Result, detail::BlockingT::NON_BLOCKING));
  // Wait will cleanup Cmd1 - not able to use any more, but Cmd2 should not be
  // cleaned up yet since we disable post enqueue cleanup.
  Cmd1Event->wait(Cmd1Event);
  EXPECT_EQ(Cmd1Event->get_info<info::event::command_execution_status>(),
            info::event_command_status::complete);
  {
    // Write lock allows to wait till all actions on host task completion are
    // executed including blocked users enqueue
    auto Lock = MS.acquireOriginSchedGraphWriteLock();
    Lock.lock();
    EXPECT_TRUE(Cmd2->isSuccessfullyEnqueued());
    EXPECT_TRUE(Cmd3->isSuccessfullyEnqueued());
  }
  Cmd3Event->wait(Cmd2Event);
}