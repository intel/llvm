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
    default:
      throw sycl::runtime_error("Unhandled type of command group",
                                PI_INVALID_OPERATION);
    }

    return CommandGroup;
  }
};

detail::Command *AddHostTaskCG(MockScheduler &MS, detail::QueueImplPtr DevQueue,
                               const std::vector<EventImplPtr> &Events) {
  std::vector<detail::Requirement *> Requirements;
  std::unique_ptr<detail::CG> CommandGroup;
  std::vector<detail::Command *> ToEnqueue;
  auto EmptyLambda = []() {};
  std::unique_ptr<detail::HostTask> HostTaskEmpty(
      new detail::HostTask(EmptyLambda));
  CommandGroup.reset(new detail::CGHostTask(
      std::move(HostTaskEmpty) /*MHostTask*/, DevQueue,
      DevQueue->getContextImplPtr(), {} /*MArgs*/, {} /*MArgsStorage*/,
      {} /*MAccStorage*/, {} /*MSharedPtrStorage*/, Requirements, Events,
      detail::CG::CodeplayHostTask, {} /*MCodeLoc*/));
  detail::Command *NewCmd =
      MS.addCG(std::move(CommandGroup), MS.getDefaultHostQueue(), ToEnqueue);
  EXPECT_EQ(ToEnqueue.size(), 0u);
  return NewCmd;
}

detail::Command *AddSingleTaskCG(context Ctx, MockScheduler &MS,
                                 detail::QueueImplPtr DevQueue,
                                 const std::vector<EventImplPtr> &Events) {
  std::vector<detail::Command *> ToEnqueue;

  kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx);
  auto ExecBundle = sycl::build(KernelBundle);

  // Emulating processing of command group function
  MockHandlerEnqueueHostDeps MockCGH(DevQueue, false);
  MockCGH.use_kernel_bundle(ExecBundle);
  for (auto EventImpl : Events)
    MockCGH.depends_on(detail::createSyclObjFromImpl<event>(EventImpl));

  MockCGH.single_task<TestKernel>([] {}); // Actual kernel does not matter

  std::unique_ptr<sycl::detail::CG> CmdGroup = MockCGH.finalize();

  detail::Command *NewCmd = MS.addCG(std::move(CmdGroup), DevQueue, ToEnqueue);
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

TEST_F(SchedulerTest, EnqueueWithHostTaskDeps1) {
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
  context Ctx{Plt};
  queue QueueDev(Ctx, Selector);
  MockScheduler MS;

  detail::QueueImplPtr QueueDevImpl = detail::getSyclObjImpl(QueueDev);
  detail::QueueImplPtr QueueHostImpl = MS.getDefaultHostQueue();

  std::vector<EventImplPtr> Events;
  std::vector<EventImplPtr> BlockingTaskEvents;

  detail::Command *Cmd1 = AddHostTaskCG(MS, QueueDevImpl, Events);
  EventImplPtr Cmd1Event = Cmd1->getEvent();
  VerifyNewHostTaskValidness(Cmd1, BlockingTaskEvents);

  // Simulate depends_on() call
  Events.push_back(Cmd1Event);
  BlockingTaskEvents.push_back(Cmd1->getEvent()->getEmptyCommandEvent());
  detail::Command *Cmd2 = AddHostTaskCG(MS, QueueDevImpl, Events);
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

TEST_F(SchedulerTest, EnqueueWithHostTaskDeps2) {
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

  context Ctx{Plt};
  queue QueueDev(Ctx, Selector);
  MockScheduler MS;

  detail::QueueImplPtr QueueDevImpl = detail::getSyclObjImpl(QueueDev);
  detail::QueueImplPtr QueueHostImpl = MS.getDefaultHostQueue();

  std::vector<EventImplPtr> Events;
  std::vector<EventImplPtr> BlockingTaskEvents;

  detail::Command *Cmd1 = AddHostTaskCG(MS, QueueDevImpl, Events);
  EventImplPtr Cmd1Event = Cmd1->getEvent();
  VerifyNewHostTaskValidness(Cmd1, BlockingTaskEvents);

  // Simulate depends_on() call
  Events.push_back(Cmd1Event);
  BlockingTaskEvents.push_back(Cmd1->getEvent()->getEmptyCommandEvent());
  detail::Command *Cmd2 = AddSingleTaskCG(Ctx, MS, QueueDevImpl, Events);
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

// TEST_F(SchedulerTest, EnqueueHostTaskDependingOnKernel)
// {
//   // Checks enqueue of host task depending on kernel
//     unittest::ScopedEnvVar DisabledCleanup{
//       DisablePostEnqueueCleanupName, "1",
//       detail::SYCLConfig<detail::SYCL_DISABLE_POST_ENQUEUE_CLEANUP>::reset};

//   default_selector Selector;
//   platform Plt{Selector};
//   if (Plt.is_host()) {
//     std::cout << "Not run due to host-only environment\n";
//     return;
//   }
//   MockScheduler MS;

//   queue QueueDev(Selector);
//   detail::QueueImplPtr QueueDevImpl = detail::getSyclObjImpl(QueueDev);
//   detail::QueueImplPtr QueueHostImpl = MS.getDefaultHostQueue();

//   std::vector<EventImplPtr> Events;
//   std::vector<EventImplPtr> BlockingTaskEvents;

//   detail::Command *Cmd1 = AddSingleTaskCG(MS, QueueDevImpl, Events);
//   EventImplPtr Cmd1Event = Cmd1->getEvent();
//   VerifyNewSingleTaskValidness(Cmd1, BlockingTaskEvents);

//   // Simulate depends_on() call
//   Events.push_back(Cmd1Event);
//   detail::Command *Cmd2 = AddHostTaskCG(MS, QueueDevImpl, Events);
//   EventImplPtr Cmd2Event = Cmd2->getEvent();
//   VerifyNewHostTaskValidness(Cmd2, BlockingTaskEvents);

//   unittest::PiMock Mock{Plt};
//   setupDefaultMockAPIs(Mock);

//   // Not expect any execution here!
//   detail::EnqueueResultT Result;
//   EXPECT_TRUE(
//       MS.enqueueCommand(Cmd2, Result, detail::BlockingT::NON_BLOCKING));
//   EXPECT_EQ(Result.MResult, detail::EnqueueResultT::SyclEnqueueSuccess);

//   // Preconditions for post enqueue checks
//   EXPECT_TRUE(Cmd1->isSuccessfullyEnqueued());
//   EXPECT_TRUE(Cmd2->isSuccessfullyEnqueued());
//   Cmd2Event->wait(Cmd2Event);
// }