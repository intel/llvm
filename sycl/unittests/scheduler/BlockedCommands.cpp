//==----------- BlockedCommands.cpp --- Scheduler unit tests ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

using namespace cl::sycl;

TEST_F(SchedulerTest, BlockedCommands) {
  MockCommand MockCmd(detail::getSyclObjImpl(MQueue));

  MockCmd.MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueBlocked;
  MockCmd.MIsBlockable = true;
  MockCmd.MRetVal = CL_DEVICE_PARTITION_EQUALLY;

  detail::EnqueueResultT Res;
  bool Enqueued =
      MockScheduler::enqueueCommand(&MockCmd, Res, detail::NON_BLOCKING);
  ASSERT_FALSE(Enqueued) << "Blocked command should not be enqueued\n";
  ASSERT_EQ(detail::EnqueueResultT::SyclEnqueueBlocked, Res.MResult)
      << "Result of enqueueing blocked command should be BLOCKED\n";

  MockCmd.MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueReady;
  Res.MResult = detail::EnqueueResultT::SyclEnqueueSuccess;
  MockCmd.MRetVal = CL_DEVICE_PARTITION_EQUALLY;

  Enqueued = MockScheduler::enqueueCommand(&MockCmd, Res, detail::BLOCKING);
  ASSERT_FALSE(Enqueued) << "Blocked command should not be enqueued\n";
  ASSERT_EQ(detail::EnqueueResultT::SyclEnqueueFailed, Res.MResult)
      << "The command is expected to fail to enqueue.\n";
  ASSERT_EQ(CL_DEVICE_PARTITION_EQUALLY, MockCmd.MRetVal)
      << "Expected different error code.\n";
  ASSERT_EQ(&MockCmd, Res.MCmd) << "Expected different failed command.\n";

  Res = detail::EnqueueResultT{};
  MockCmd.MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueReady;
  MockCmd.MRetVal = CL_SUCCESS;
  Enqueued = MockScheduler::enqueueCommand(&MockCmd, Res, detail::BLOCKING);
  ASSERT_TRUE(Enqueued &&
              Res.MResult == detail::EnqueueResultT::SyclEnqueueSuccess)
      << "The command is expected to be successfully enqueued.\n";
}
