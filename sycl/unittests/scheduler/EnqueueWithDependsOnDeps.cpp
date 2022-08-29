//==------------ EnqueueWithDependsOnDeps.cpp --- Scheduler unit tests------==//
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

#include <vector>

using namespace sycl;
using EventImplPtr = std::shared_ptr<detail::event_impl>;

namespace DependsOnTest {
class MockHandlerCustom : public MockHandler {
public:
  MockHandlerCustom(std::shared_ptr<sycl::detail::queue_impl> Queue,
                    bool IsHost)
      : MockHandler(Queue, IsHost) {}

  std::unique_ptr<sycl::detail::CG> finalize() {
    std::unique_ptr<sycl::detail::CG> CommandGroup;
    switch (getType()) {
    case sycl::detail::CG::Kernel: {
      CommandGroup.reset(new sycl::detail::CGExecKernel(
          getNDRDesc(), std::move(getHostKernel()), getKernel(),
          std::move(MImpl->MKernelBundle), getArgsStorage(), getAccStorage(),
          getSharedPtrStorage(), getRequirements(), getEvents(), getArgs(),
          getKernelName(), getOSModuleHandle(), getStreamStorage(),
          MImpl->MAuxiliaryResources, getCGType(), getCodeLoc()));
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
                                PI_ERROR_INVALID_OPERATION);
    }

    return CommandGroup;
  }
};
} // namespace DependsOnTest
detail::Command *AddTaskCG(bool IsHost, MockScheduler &MS,
                           detail::QueueImplPtr DevQueue,
                           const std::vector<EventImplPtr> &Events) {
  std::vector<detail::Command *> ToEnqueue;

  // Emulating processing of command group function
  DependsOnTest::MockHandlerCustom MockCGH(DevQueue, false);

  for (auto EventImpl : Events)
    MockCGH.depends_on(detail::createSyclObjFromImpl<event>(EventImpl));

  if (IsHost)
    MockCGH.host_task([] {});
  else {
    kernel_bundle KernelBundle =
        sycl::get_kernel_bundle<sycl::bundle_state::input>(
            DevQueue->get_context());
    auto ExecBundle = sycl::build(KernelBundle);
    MockCGH.use_kernel_bundle(ExecBundle);
    MockCGH.single_task<TestKernel<>>([] {});
  }

  std::unique_ptr<sycl::detail::CG> CmdGroup = MockCGH.finalize();

  detail::Command *NewCmd =
      MS.addCG(std::move(CmdGroup),
               IsHost ? MS.getDefaultHostQueue() : DevQueue, ToEnqueue);
  EXPECT_EQ(ToEnqueue.size(), 0u);
  return NewCmd;
}

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

void VerifyTaskStructureValidness(
    detail::Command *NewCmd, const std::vector<EventImplPtr> &BlockingTasks) {
  ASSERT_NE(NewCmd, nullptr);
  EXPECT_EQ(NewCmd->getType(), detail::Command::RUN_CG);

  EventImplPtr NewCmdEvent = NewCmd->getEvent();
  ASSERT_NE(NewCmdEvent, nullptr);

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
              (static_cast<detail::Command *>(BlockingTask->getCommand()))
                  ->getBlockedUsers();
          return BlockedUsers.find(NewCmdEvent) != BlockedUsers.end();
        }));
  }
}

inline constexpr auto DisablePostEnqueueCleanupName =
    "SYCL_DISABLE_POST_ENQUEUE_CLEANUP";

// TEST_F(SchedulerTest, EnqueueNoMemObjTwoHostTasks) {
//   // Checks enqueue of two dependent host tasks

//   unittest::ScopedEnvVar DisabledCleanup{
//       DisablePostEnqueueCleanupName, "1",
//       detail::SYCLConfig<detail::SYCL_DISABLE_POST_ENQUEUE_CLEANUP>::reset};

//   default_selector Selector;
//   platform Plt{Selector};
//   if (Plt.is_host()) {
//     std::cout << "Not run due to host-only environment\n";
//     return;
//   }
//   // This test only contains device image for SPIR-V capable devices.
//   if (Plt.get_backend() != sycl::backend::opencl &&
//       Plt.get_backend() != sycl::backend::ext_oneapi_level_zero) {
//     std::cout << "Only OpenCL and Level Zero are supported for this test\n";
//     return;
//   }

//   queue QueueDev(context(Plt), Selector);
//   MockScheduler MS;

//   detail::QueueImplPtr QueueDevImpl = detail::getSyclObjImpl(QueueDev);
//   detail::QueueImplPtr QueueHostImpl = MS.getDefaultHostQueue();

//   std::vector<EventImplPtr> Events;

//   detail::Command *Cmd1 = AddTaskCG(true, MS, QueueDevImpl, Events);
//   EventImplPtr Cmd1Event = Cmd1->getEvent();

//   // Simulate depends_on() call
//   Events.push_back(Cmd1Event);
//   detail::Command *Cmd2 = AddTaskCG(true, MS, QueueDevImpl, Events);
//   EventImplPtr Cmd2Event = Cmd2->getEvent();

//   detail::EnqueueResultT Result;
//   EXPECT_TRUE(MS.enqueueCommand(Cmd2, Result, detail::BlockingT::NON_BLOCKING));

//   // Preconditions for post enqueue checks
//   EXPECT_TRUE(Cmd1->isSuccessfullyEnqueued());
//   EXPECT_TRUE(Cmd2->isSuccessfullyEnqueued());

//   Cmd2Event->wait(Cmd2Event);
//   EXPECT_EQ(Cmd1Event->get_info<info::event::command_execution_status>(),
//             info::event_command_status::complete);
//   EXPECT_EQ(Cmd2Event->get_info<info::event::command_execution_status>(),
//             info::event_command_status::complete);
// }

TEST_F(SchedulerTest, EnqueueNoMemObjKernelDepHost) {
  // Checks enqueue of kernel depending on host task
  unittest::ScopedEnvVar DisabledCleanup{
      DisablePostEnqueueCleanupName, "1",
      detail::SYCLConfig<detail::SYCL_DISABLE_POST_ENQUEUE_CLEANUP>::reset};

  default_selector Selector;
  platform Plt{Selector};

  if (!CheckTestExecutionRequirements(Plt))
    return;

  unittest::PiMock Mock{Plt};
  setupDefaultMockAPIs(Mock);

  queue QueueDev(context(Plt), Selector);
  MockScheduler MS;

  detail::QueueImplPtr QueueDevImpl = detail::getSyclObjImpl(QueueDev);

  std::vector<EventImplPtr> Events;
  std::vector<EventImplPtr> BlockingTaskEvents;

  detail::Command *Cmd1 = AddTaskCG(true, MS, QueueDevImpl, Events);
  EventImplPtr Cmd1Event = Cmd1->getEvent();
  VerifyTaskStructureValidness(Cmd1, BlockingTaskEvents);

  // Simulate depends_on() call
  Events.push_back(Cmd1Event);
  BlockingTaskEvents.push_back(Cmd1->getEvent());
  detail::Command *Cmd2 = AddTaskCG(false, MS, QueueDevImpl, Events);
  EventImplPtr Cmd2Event = Cmd2->getEvent();
  VerifyTaskStructureValidness(Cmd2, BlockingTaskEvents);

  detail::EnqueueResultT Result;
  EXPECT_FALSE(MS.enqueueCommand(Cmd2, Result, detail::BlockingT::NON_BLOCKING));

  // Preconditions for post enqueue checks
  EXPECT_TRUE(Cmd1->isSuccessfullyEnqueued());
  EXPECT_FALSE(Cmd2->isSuccessfullyEnqueued());

  EXPECT_EQ(Cmd1Event->get_info<info::event::command_execution_status>(),
            info::event_command_status::submitted);
  EXPECT_EQ(Cmd2Event->get_info<info::event::command_execution_status>(),
            info::event_command_status::submitted);
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

  if (!CheckTestExecutionRequirements(Plt))
    return;

  unittest::PiMock Mock{Plt};
  setupDefaultMockAPIs(Mock);

  queue QueueDev(context(Plt), Selector);
  MockScheduler MS;

  detail::QueueImplPtr QueueDevImpl = detail::getSyclObjImpl(QueueDev);

  std::vector<EventImplPtr> Events;
  std::vector<EventImplPtr> BlockingTaskEvents;

  detail::Command *Cmd1 = AddTaskCG(false, MS, QueueDevImpl, Events);
  EventImplPtr Cmd1Event = Cmd1->getEvent();
  VerifyTaskStructureValidness(Cmd1, BlockingTaskEvents);

  // Simulate depends_on() call
  Events.push_back(Cmd1Event);
  BlockingTaskEvents.clear();
  detail::Command *Cmd2 = AddTaskCG(true, MS, QueueDevImpl, Events);
  EventImplPtr Cmd2Event = Cmd2->getEvent();
  VerifyTaskStructureValidness(Cmd2, BlockingTaskEvents);

  detail::EnqueueResultT Result;
  EXPECT_TRUE(MS.enqueueCommand(Cmd2, Result, detail::BlockingT::NON_BLOCKING));

  // Preconditions for post enqueue checks
  EXPECT_TRUE(Cmd1->isSuccessfullyEnqueued());
  EXPECT_TRUE(Cmd2->isSuccessfullyEnqueued());
  Cmd2Event->wait(Cmd2Event);
  EXPECT_EQ(Cmd2Event->get_info<info::event::command_execution_status>(),
            info::event_command_status::complete);
  // Not enable check below since no kernel execution involved
  // EXPECT_EQ(Cmd1Event->get_info<info::event::command_execution_status>(),
  //           info::event_command_status::complete);
}

TEST_F(SchedulerTest, EnqueueNoMemObjDoubleKernelDepHostBlocked) {
  // Checks blocking command tranfer for dependent kernels and enqueue of root
  // kernel on host task completion
  unittest::ScopedEnvVar DisabledCleanup{
      DisablePostEnqueueCleanupName, "1",
      detail::SYCLConfig<detail::SYCL_DISABLE_POST_ENQUEUE_CLEANUP>::reset};

  default_selector Selector;
  platform Plt{Selector};

  if (!CheckTestExecutionRequirements(Plt))
    return;

  unittest::PiMock Mock{Plt};
  setupDefaultMockAPIs(Mock);

  queue QueueDev(context(Plt), Selector);
  MockScheduler MS;

  detail::QueueImplPtr QueueDevImpl = detail::getSyclObjImpl(QueueDev);

  std::vector<EventImplPtr> Events;
  std::vector<EventImplPtr> BlockingTaskEvents;

  detail::Command *Cmd1 = AddTaskCG(true, MS, QueueDevImpl, Events);
  EventImplPtr Cmd1Event = Cmd1->getEvent();
  VerifyTaskStructureValidness(Cmd1, BlockingTaskEvents);
  // No such real usage but we need to postpone cmd1 enqueue for checks
  Cmd1->MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueBlocked;

  // Depends on host task
  Events.push_back(Cmd1Event);
  BlockingTaskEvents.push_back(Cmd1Event);
  detail::Command *Cmd2 = AddTaskCG(false, MS, QueueDevImpl, Events);
  EventImplPtr Cmd2Event = Cmd2->getEvent();
  VerifyTaskStructureValidness(Cmd2, BlockingTaskEvents);

  // Depends on kernel depending on host task
  Events.clear();
  Events.push_back(Cmd2Event);
  // We expect to see host task as blocking task for second layer kernel
  ASSERT_EQ(BlockingTaskEvents.size(), 1u);
  detail::Command *Cmd3 = AddTaskCG(false, MS, QueueDevImpl, Events);
  EventImplPtr Cmd3Event = Cmd2->getEvent();
  VerifyTaskStructureValidness(Cmd3, BlockingTaskEvents);

  detail::EnqueueResultT Result;
  EXPECT_FALSE(
      MS.enqueueCommand(Cmd2, Result, detail::BlockingT::NON_BLOCKING));
  EXPECT_EQ(Result.MResult, detail::EnqueueResultT::SyclEnqueueBlocked);
  EXPECT_EQ(Result.MCmd, static_cast<detail::Command *>(Cmd1));

  EXPECT_FALSE(Cmd1->isSuccessfullyEnqueued());
  EXPECT_FALSE(Cmd2->isSuccessfullyEnqueued());
  EXPECT_EQ(Cmd1Event->get_info<info::event::command_execution_status>(),
          info::event_command_status::submitted);

  EXPECT_FALSE(
      MS.enqueueCommand(Cmd3, Result, detail::BlockingT::NON_BLOCKING));
  EXPECT_EQ(Result.MResult, detail::EnqueueResultT::SyclEnqueueBlocked);
  EXPECT_EQ(Result.MCmd, static_cast<detail::Command *>(Cmd1));

  EXPECT_FALSE(Cmd1->isSuccessfullyEnqueued());
  EXPECT_FALSE(Cmd2->isSuccessfullyEnqueued());
  EXPECT_FALSE(Cmd3->isSuccessfullyEnqueued());

  Cmd1->MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueReady;

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

std::vector<std::pair<pi_uint32, const pi_event *>> PassedNumEvents;
inline pi_result redefinedEventsWaitCustom(pi_uint32 num_events,
                                           const pi_event *event_list) {
  PassedNumEvents.push_back(std::make_pair(num_events, event_list));
  return PI_SUCCESS;
}

std::vector<std::pair<pi_uint32, const pi_event *>> PassedNumEventsToLaunch;
inline pi_result redefinedEnqueueKernelLaunchCustom(
    pi_queue, pi_kernel, pi_uint32, const size_t *, const size_t *,
    const size_t *, pi_uint32 num_events, const pi_event *event_list,
    pi_event *event) {
  static int eventCounter = 0x00;
  PassedNumEventsToLaunch.push_back(std::make_pair(num_events, event_list));
  *event = reinterpret_cast<pi_event>(new int{++eventCounter});

  return PI_SUCCESS;
}

void EventsWaitVerification(queue &QueueDev) {
  MockScheduler MS;

  detail::QueueImplPtr QueueDevImpl = detail::getSyclObjImpl(QueueDev);

  std::vector<EventImplPtr> Events;

  detail::Command *Cmd1 = AddTaskCG(true, MS, QueueDevImpl, Events);
  EventImplPtr Cmd1Event = Cmd1->getEvent();

  // Depends on host task
  Events.push_back(Cmd1Event);
  detail::Command *Cmd2 = AddTaskCG(false, MS, QueueDevImpl, Events);
  EventImplPtr Cmd2Event = Cmd2->getEvent();

  // Depends on kernel depending on host task
  Events.clear();
  Events.push_back(Cmd2Event);
  detail::Command *Cmd3 = AddTaskCG(false, MS, QueueDevImpl, Events);
  EventImplPtr Cmd3Event = Cmd2->getEvent();

  detail::EnqueueResultT Result;
  MS.enqueueCommand(Cmd3, Result, detail::BlockingT::NON_BLOCKING);
  EXPECT_TRUE(Cmd1->isSuccessfullyEnqueued());
  Cmd3Event->wait(Cmd3Event);

  // One piEventsWait call:
  // kernel2 waits for kernel 1 by sending event list to enqueue launch call
  // (depending on queue property). Cmd3Event.wait() waits for kernel2 via
  // piEventsWait.
  ASSERT_EQ(PassedNumEvents.size(), 1u);
  auto [EventCount, EventArr] = PassedNumEvents[0];
  ASSERT_EQ(EventCount, 1u);
  EXPECT_EQ(*EventArr, Cmd3Event->getHandleRef());
}

TEST_F(SchedulerTest, InOrderEnqueueNoMemObjDoubleKernelDepHost) {
  // Checks blocking command tranfer for dependent kernels and enqueue of root
  // kernel on host task completion
  unittest::ScopedEnvVar DisabledCleanup{
      DisablePostEnqueueCleanupName, "1",
      detail::SYCLConfig<detail::SYCL_DISABLE_POST_ENQUEUE_CLEANUP>::reset};

  default_selector Selector;
  platform Plt{Selector};

  if (!CheckTestExecutionRequirements(Plt))
    return;

  unittest::PiMock Mock{Plt};
  setupDefaultMockAPIs(Mock);
  Mock.redefine<detail::PiApiKind::piEventsWait>(redefinedEventsWaitCustom);
  Mock.redefine<detail::PiApiKind::piEnqueueKernelLaunch>(
      redefinedEnqueueKernelLaunchCustom);

  {
    queue QueueDev(context(Plt), Selector);
    PassedNumEvents.clear();
    PassedNumEventsToLaunch.clear();
    EventsWaitVerification(QueueDev);
    // 1st -> kernel after host, no pi events
    // 2nd -> kernel after kernel, 1 pi event
    ASSERT_EQ(PassedNumEventsToLaunch.size(), 2u);
    {
      auto [EventCount, EventArr] = PassedNumEventsToLaunch[0];
      EXPECT_EQ(EventCount, 0u);
      EXPECT_EQ(EventArr, nullptr);
    }
    {
      auto [EventCount, EventArr] = PassedNumEventsToLaunch[1];
      EXPECT_EQ(EventCount, 1u);
    }
  }

  {
    queue QueueDev(context(Plt), Selector, property::queue::in_order());
    PassedNumEvents.clear();
    PassedNumEventsToLaunch.clear();
    EventsWaitVerification(QueueDev);
    // 1st -> kernel after host, no pi events
    // 2nd -> kernel after kernel and in order queue, 0 pi event
    ASSERT_EQ(PassedNumEventsToLaunch.size(), 2u);
    {
      auto [EventCount, EventArr] = PassedNumEventsToLaunch[0];
      EXPECT_EQ(EventCount, 0u);
      EXPECT_EQ(EventArr, nullptr);
    }
    {
      auto [EventCount, EventArr] = PassedNumEventsToLaunch[1];
      EXPECT_EQ(EventCount, 0u);
      EXPECT_EQ(EventArr, nullptr);
    }
  }
}