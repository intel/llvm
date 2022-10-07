// //==------------ EnqueueWithDependsOnDeps.cpp --- Scheduler unit
// tests------==//
// //
// // Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// // See https://llvm.org/LICENSE.txt for license information.
// // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// //
// //===----------------------------------------------------------------------===//

// #include "SchedulerTest.hpp"
// #include "SchedulerTestUtils.hpp"
// #include <detail/handler_impl.hpp>

// #include <helpers/PiMock.hpp>
// #include <helpers/ScopedEnvVar.hpp>
// #include <helpers/TestKernel.hpp>

// #include <vector>

// using namespace sycl;
// using EventImplPtr = std::shared_ptr<detail::event_impl>;

// namespace DependsOnTest {
// class MockHandlerCustom : public MockHandler {
// public:
//   MockHandlerCustom(std::shared_ptr<sycl::detail::queue_impl> Queue,
//                     bool IsHost)
//       : MockHandler(Queue, IsHost) {}

//   std::unique_ptr<sycl::detail::CG> finalize() {
//     std::unique_ptr<sycl::detail::CG> CommandGroup;
//     switch (getType()) {
//     case sycl::detail::CG::Kernel: {
//       CommandGroup.reset(new sycl::detail::CGExecKernel(
//           getNDRDesc(), std::move(getHostKernel()), getKernel(),
//           std::move(MImpl->MKernelBundle), getArgsStorage(), getAccStorage(),
//           getSharedPtrStorage(), getRequirements(), getEvents(), getArgs(),
//           getKernelName(), getOSModuleHandle(), getStreamStorage(),
//           MImpl->MAuxiliaryResources, getCGType(), getCodeLoc()));
//       break;
//     }
//     case sycl::detail::CG::CodeplayHostTask: {
//       CommandGroup.reset(new detail::CGHostTask(
//           std::move(getHostTask()), getQueue(),
//           getQueue()->getContextImplPtr(), getArgs(), getArgsStorage(),
//           getAccStorage(), getSharedPtrStorage(), getRequirements(),
//           getEvents(), getCGType(), getCodeLoc()));
//       break;
//     }
//     default:
//       throw sycl::runtime_error("Unhandled type of command group",
//                                 PI_ERROR_INVALID_OPERATION);
//     }

//     return CommandGroup;
//   }
// };
// } // namespace DependsOnTest

// enum TestCGType
// {
//   KERNEL_TASK = 0x00,
//   HOST_TASK   = 0x01
// };

// detail::Command *AddTaskCG(TestCGType Type, MockScheduler &MS,
//                            detail::QueueImplPtr DevQueue,
//                            const std::vector<EventImplPtr> &Events) {
//   std::vector<detail::Command *> ToEnqueue;

//   // Emulating processing of command group function
//   DependsOnTest::MockHandlerCustom MockCGH(DevQueue, false);

//   for (auto EventImpl : Events)
//     MockCGH.depends_on(detail::createSyclObjFromImpl<event>(EventImpl));

//   if (Type == TestCGType::HOST_TASK)
//     MockCGH.host_task([] {});
//   else {
//     kernel_bundle KernelBundle =
//         sycl::get_kernel_bundle<sycl::bundle_state::input>(
//             DevQueue->get_context());
//     auto ExecBundle = sycl::build(KernelBundle);
//     MockCGH.use_kernel_bundle(ExecBundle);
//     MockCGH.single_task<TestKernel<>>([] {});
//   }

//   std::unique_ptr<sycl::detail::CG> CmdGroup = MockCGH.finalize();

//   detail::Command *NewCmd =
//       MS.addCG(std::move(CmdGroup),
//                Type == TestCGType::HOST_TASK ? MS.getDefaultHostQueue() :
//                DevQueue, ToEnqueue);
//   EXPECT_EQ(ToEnqueue.size(), 0u);
//   return NewCmd;
// }

// bool CheckTestExecutionRequirements(const platform &plt) {
//   if (plt.is_host()) {
//     std::cout << "Not run due to host-only environment\n";
//     return false;
//   }
//   // This test only contains device image for SPIR-V capable devices.
//   if (plt.get_backend() != sycl::backend::opencl &&
//       plt.get_backend() != sycl::backend::ext_oneapi_level_zero) {
//     std::cout << "Only OpenCL and Level Zero are supported for this test\n";
//     return false;
//   }
//   return true;
// }

// void VerifyTaskStructureValidness(
//     detail::Command *NewCmd, const std::vector<detail::Command*>
//     &BlockingTasks) {
//   ASSERT_NE(NewCmd, nullptr);
//   EXPECT_EQ(NewCmd->getType(), detail::Command::RUN_CG);

//   EXPECT_TRUE(std::all_of(
//       BlockingTasks.cbegin(), BlockingTasks.cend(),
//       [&NewCmd](detail::Command* BlockingTask) {
//         const auto &BlockedUsers = BlockingTask->getBlockedUsers();
//         return std::find(BlockedUsers.begin(), BlockedUsers.end(), NewCmd) !=
//         BlockedUsers.end();
//       }));
// }

// inline constexpr auto DisablePostEnqueueCleanupName =
//     "SYCL_DISABLE_POST_ENQUEUE_CLEANUP";

// TEST_F(SchedulerTest, EnqueueNoMemObjKernelDepHost) {
//   // Checks enqueue of kernel depending on host task
//   unittest::ScopedEnvVar DisabledCleanup{
//       DisablePostEnqueueCleanupName, "1",
//       detail::SYCLConfig<detail::SYCL_DISABLE_POST_ENQUEUE_CLEANUP>::reset};

//   default_selector Selector;
//   platform Plt{Selector};

//   if (!CheckTestExecutionRequirements(Plt))
//     GTEST_SKIP();

//   sycl::unittest::PiMock Mock;

//   queue QueueDev(context(Plt), Selector);
//   MockScheduler MS;

//   detail::QueueImplPtr QueueDevImpl = detail::getSyclObjImpl(QueueDev);

//   std::vector<EventImplPtr> Events;
//   std::vector<detail::Command*> BlockingTasks;

//   detail::Command *Cmd1 = AddTaskCG(TestCGType::HOST_TASK, MS, QueueDevImpl,
//   Events); EventImplPtr Cmd1Event = Cmd1->getEvent();
//   VerifyTaskStructureValidness(Cmd1, BlockingTasks);

//   // Simulate depends_on() call
//   Events.push_back(Cmd1Event);
//   BlockingTasks.push_back(Cmd1);
//   detail::Command *Cmd2 = AddTaskCG(TestCGType::KERNEL_TASK, MS,
//   QueueDevImpl, Events); EventImplPtr Cmd2Event = Cmd2->getEvent();

//   detail::EnqueueResultT Result;
//   EXPECT_FALSE(MS.enqueueCommand(Cmd2, Result,
//   detail::BlockingT::NON_BLOCKING)); VerifyTaskStructureValidness(Cmd2,
//   BlockingTasks);

//   // Preconditions for post enqueue checks
//   EXPECT_TRUE(Cmd1->isSuccessfullyEnqueued());
//   EXPECT_FALSE(Cmd2->isSuccessfullyEnqueued());

//   EXPECT_EQ(Cmd1Event->get_info<info::event::command_execution_status>(),
//             info::event_command_status::submitted);
//   EXPECT_EQ(Cmd2Event->get_info<info::event::command_execution_status>(),
//             info::event_command_status::submitted);
//   // Wait will cleanup Cmd1 - not able to use any more, but Cmd2 should not
//   be
//   // cleaned up yet since we disable post enqueue cleanup.
//   Cmd1Event->wait(Cmd1Event);
//   EXPECT_EQ(Cmd1Event->get_info<info::event::command_execution_status>(),
//             info::event_command_status::complete);
//   {
//     // Write lock allows to wait till all actions on host task completion are
//     // executed including blocked users enqueue
//     auto Lock = MS.acquireOriginSchedGraphWriteLock();
//     Lock.lock();
//     EXPECT_TRUE(Cmd2->isSuccessfullyEnqueued());
//   }

//   Cmd2Event->wait(Cmd2Event);
// }