//==----------- BlockedCommands.cpp --- Scheduler unit tests ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"

#include <CL/cl.h>
#include <CL/sycl.hpp>
#include <detail/scheduler/scheduler.hpp>

#include <gtest/gtest.h>

using namespace cl::sycl;

class MockCommand : public detail::Command {
public:
  MockCommand(detail::QueueImplPtr Queue)
      : Command(detail::Command::ALLOCA, Queue) {}
  void printDot(std::ostream &Stream) const override {}

  void emitInstrumentationData() override {}

  cl_int enqueueImp() override { return MRetVal; }

  cl_int MRetVal = CL_SUCCESS;
};

class TestScheduler : public detail::Scheduler {
public:
  static bool enqueueCommand(detail::Command *Cmd,
                             detail::EnqueueResultT &EnqueueResult,
                             detail::BlockingT Blocking) {
    return GraphProcessor::enqueueCommand(Cmd, EnqueueResult, Blocking);
  }
};

TEST_F(SchedulerTest, BlockedCommands) {
  MockCommand MockCmd(detail::getSyclObjImpl(MQueue));

  MockCmd.MIsBlockable = true;
  MockCmd.MCanEnqueue = false;
  MockCmd.MRetVal = CL_DEVICE_PARTITION_EQUALLY;

  detail::EnqueueResultT Res;
  bool Enqueued =
      TestScheduler::enqueueCommand(&MockCmd, Res, detail::NON_BLOCKING);
  ASSERT_FALSE(Enqueued) << "Blocked command should not be enqueued\n";
  ASSERT_EQ(detail::EnqueueResultT::SyclEnqueueBlocked, Res.MResult)
      << "Result of enqueueing blocked command should be BLOCKED\n";

  MockCmd.MCanEnqueue = true;
  Res.MResult = detail::EnqueueResultT::SyclEnqueueSuccess;
  MockCmd.MRetVal = CL_DEVICE_PARTITION_EQUALLY;

  Enqueued = TestScheduler::enqueueCommand(&MockCmd, Res, detail::BLOCKING);
  ASSERT_FALSE(Enqueued) << "Blocked command should not be enqueued\n";
  ASSERT_EQ(detail::EnqueueResultT::SyclEnqueueFailed, Res.MResult)
      << "The command is expected to fail to enqueue.\n";
  ASSERT_EQ(CL_DEVICE_PARTITION_EQUALLY, MockCmd.MRetVal)
      << "Expected different error code.\n";
  ASSERT_EQ(&MockCmd, Res.MCmd) << "Expected different failed command.\n";

  Res = detail::EnqueueResultT{};
  MockCmd.MRetVal = CL_SUCCESS;
  Enqueued = TestScheduler::enqueueCommand(&MockCmd, Res, detail::BLOCKING);
  ASSERT_TRUE(Enqueued &&
              Res.MResult == detail::EnqueueResultT::SyclEnqueueSuccess)
      << "The command is expected to be successfully enqueued.\n";
}
