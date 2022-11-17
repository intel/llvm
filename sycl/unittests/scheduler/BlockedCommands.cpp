//==----------- BlockedCommands.cpp --- Scheduler unit tests ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

#include <helpers/PiMock.hpp>

using namespace sycl;
using namespace testing;

TEST_F(SchedulerTest, BlockedCommands) {
  sycl::unittest::PiMock Mock;
  sycl::queue Q{Mock.getPlatform().get_devices()[0], MAsyncHandler};
  MockCommand MockCmdBlocking(detail::getSyclObjImpl(Q));
  MockCommand MockCmd(detail::getSyclObjImpl(Q));

  EXPECT_TRUE(MockCmdBlocking.blockManually(
      detail::Command::BlockReason::HostAccessor));
  EXPECT_TRUE(MockCmdBlocking.isBlocking());
  std::vector<detail::Command *> ToCleanUp;
  std::ignore = MockCmd.addDep(MockCmdBlocking.getEvent(), ToCleanUp);

  MockScheduler MS;
  auto Lock = MS.acquireGraphReadLock();
  detail::EnqueueResultT Res;
  bool Enqueued =
      MockScheduler::enqueueCommand(&MockCmd, Res, detail::NON_BLOCKING);
  ASSERT_FALSE(Enqueued) << "Blocked command should not be enqueued\n";
  ASSERT_EQ(detail::EnqueueResultT::SyclEnqueueBlocked, Res.MResult)
      << "Result of enqueueing blocked command should be BLOCKED\n";

  Res = detail::EnqueueResultT{};
  MockCmdBlocking.unblock();
  Enqueued = MockScheduler::enqueueCommand(&MockCmd, Res, detail::NON_BLOCKING);
  ASSERT_TRUE(Enqueued &&
              Res.MResult == detail::EnqueueResultT::SyclEnqueueSuccess)
      << "The command is expected to be successfully enqueued.\n";
}

TEST_F(SchedulerTest, DontEnqueueDepsIfOneOfThemIsBlocked) {
  sycl::unittest::PiMock Mock;
  sycl::queue Q{Mock.getPlatform().get_devices()[0], MAsyncHandler};

  MockCommand A(detail::getSyclObjImpl(Q));
  A.MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueReady;
  A.MRetVal = CL_SUCCESS;

  MockCommand B(detail::getSyclObjImpl(Q));
  B.MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueReady;
  B.MRetVal = CL_SUCCESS;

  MockCommand C(detail::getSyclObjImpl(Q));
  C.MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueReady;
  C.blockManually(detail::Command::BlockReason::HostAccessor);

  MockCommand D(detail::getSyclObjImpl(Q));
  D.MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueReady;
  D.MRetVal = CL_SUCCESS;

  addEdge(&A, &B, nullptr);
  addEdge(&A, &C, nullptr);
  addEdge(&A, &D, nullptr);

  // We have such a graph:
  //
  //     A
  //   / | \
  //  B  C  D
  //
  // If C is blocked, we should not try to enqueue D.

  EXPECT_CALL(A, enqueue).Times(0);
  EXPECT_CALL(B, enqueue).Times(1);
  EXPECT_CALL(C, enqueue).Times(1);
  EXPECT_CALL(D, enqueue).Times(0);

  MockScheduler MS;
  auto Lock = MS.acquireGraphReadLock();
  detail::EnqueueResultT Res;
  bool Enqueued = MockScheduler::enqueueCommand(&A, Res, detail::NON_BLOCKING);
  ASSERT_FALSE(Enqueued) << "Blocked command should not be enqueued\n";
  ASSERT_EQ(detail::EnqueueResultT::SyclEnqueueBlocked, Res.MResult)
      << "Result of enqueueing blocked command should be BLOCKED.\n";
  ASSERT_EQ(&C, Res.MCmd) << "Expected different failed command.\n";
}

TEST_F(SchedulerTest, EnqueueBlockedCommandNoEarlyExit) {
  sycl::unittest::PiMock Mock;
  sycl::queue Q{Mock.getPlatform().get_devices()[0], MAsyncHandler};

  MockCommand A(detail::getSyclObjImpl(Q));
  A.blockManually(detail::Command::BlockReason::HostAccessor);

  MockCommand B(detail::getSyclObjImpl(Q));
  B.MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueReady;
  B.MRetVal = CL_SUCCESS;

  addEdge(&A, &B, nullptr);

  // We have such a graph:
  //
  //   A -> B
  //
  // If A is blocked, we still should try to enqueue B.

  EXPECT_CALL(A, enqueue).Times(1);
  EXPECT_CALL(B, enqueue).Times(1);

  MockScheduler MS;
  auto Lock = MS.acquireGraphReadLock();
  detail::EnqueueResultT Res;
  bool Enqueued = MockScheduler::enqueueCommand(&A, Res, detail::NON_BLOCKING);
  ASSERT_TRUE(Enqueued) << "Blocking command prevent user from being enqueued "
                           "but could be enqueued itself\n";
}

// This unit test is for workaround described in GraphProcessor::enqueueCommand
// method.
TEST_F(SchedulerTest, EnqueueHostDependency) {
  sycl::unittest::PiMock Mock;
  sycl::queue Q{Mock.getPlatform().get_devices()[0], MAsyncHandler};

  MockCommand A(detail::getSyclObjImpl(Q));
  A.MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueReady;
  A.MRetVal = CL_SUCCESS;

  MockCommand B(detail::getSyclObjImpl(Q));
  B.MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueReady;
  B.MRetVal = CL_SUCCESS;

  sycl::detail::EventImplPtr DepEvent{
      new sycl::detail::event_impl(detail::getSyclObjImpl(Q))};
  DepEvent->setCommand(&B);

  std::vector<detail::Command *> ToCleanUp;
  (void)A.addDep(DepEvent, ToCleanUp);

  // We have such a "graph":
  //
  //     A
  //     |
  //     B
  //
  // A depends on B. B is host command.
  // "Graph" is quoted as we don't have this dependency in MDeps. Instead, we
  // have this dependecy as result of handler::depends_on() call.

  EXPECT_CALL(A, enqueue).Times(1);
  EXPECT_CALL(B, enqueue).Times(1);

  MockScheduler MS;
  auto Lock = MS.acquireGraphReadLock();
  detail::EnqueueResultT Res;
  bool Enqueued = MockScheduler::enqueueCommand(&A, Res, detail::NON_BLOCKING);
  ASSERT_TRUE(Enqueued) << "The command should be enqueued\n";
  ASSERT_EQ(detail::EnqueueResultT::SyclEnqueueSuccess, Res.MResult)
      << "Enqueue operation should return successfully.\n";
}
