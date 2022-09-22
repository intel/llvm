//==------------ WaitAfterCleanup.cpp ---- Scheduler unit tests ------------==//
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

using namespace sycl;

TEST_F(SchedulerTest, PostEnqueueCleanupForCommandDefault) {
  sycl::unittest::PiMock Mock;
  sycl::queue Q{Mock.getPlatform().get_devices()[0], MAsyncHandler};

  auto Cmd = new MockCommand(detail::getSyclObjImpl(Q));
  auto Event = Cmd->getEvent();
  ASSERT_FALSE(Event == nullptr) << "Command must have an event\n";

  detail::Scheduler::getInstance().waitForEvent(Event);
  EXPECT_EQ(Event->getCommand(), nullptr) << "Command should be cleaned up\n";
}

TEST_F(SchedulerTest, WaitAfterCleanup) {
  unittest::ScopedEnvVar DisabledCleanup{
      "SYCL_DISABLE_POST_ENQUEUE_CLEANUP", "1",
      detail::SYCLConfig<detail::SYCL_DISABLE_POST_ENQUEUE_CLEANUP>::reset};

  sycl::unittest::PiMock Mock;
  sycl::queue Q{Mock.getPlatform().get_devices()[0], MAsyncHandler};

  auto Cmd = new MockCommand(detail::getSyclObjImpl(Q));
  auto Event = Cmd->getEvent();
  ASSERT_FALSE(Event == nullptr) << "Command must have an event\n";

  detail::Scheduler::getInstance().waitForEvent(Event);
  ASSERT_EQ(Event->getCommand(), Cmd)
      << "Command should not have been cleaned up yet\n";

  detail::Scheduler::getInstance().cleanupFinishedCommands(Event);
  ASSERT_TRUE(Event->getCommand() == nullptr)
      << "Command should have been cleaned up\n";

  detail::Scheduler::getInstance().waitForEvent(Event);
}
