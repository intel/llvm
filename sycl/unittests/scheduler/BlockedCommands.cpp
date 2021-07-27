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
using namespace testing;

TEST_F(SchedulerTest, BlockedCommands) {
  MockCommand MockCmd(detail::getSyclObjImpl(MQueue));

  MockCmd.MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueBlocked;
  MockCmd.MIsBlockable = true;
  MockCmd.MRetVal = CL_DEVICE_PARTITION_EQUALLY;

  MockScheduler MS;
  auto Lock = MS.acquireGraphReadLock();
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

TEST_F(SchedulerTest, DontEnqueueDepsIfOneOfThemIsBlocked) {
  MockCommand A(detail::getSyclObjImpl(MQueue));
  A.MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueReady;
  A.MIsBlockable = true;
  A.MRetVal = CL_SUCCESS;

  MockCommand B(detail::getSyclObjImpl(MQueue));
  B.MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueReady;
  B.MIsBlockable = true;
  B.MRetVal = CL_SUCCESS;

  MockCommand C(detail::getSyclObjImpl(MQueue));
  C.MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueBlocked;
  C.MIsBlockable = true;

  MockCommand D(detail::getSyclObjImpl(MQueue));
  D.MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueReady;
  D.MIsBlockable = true;
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

  EXPECT_CALL(A, enqueue(_, _)).Times(0);
  EXPECT_CALL(B, enqueue(_, _)).Times(1);
  EXPECT_CALL(C, enqueue(_, _)).Times(0);
  EXPECT_CALL(D, enqueue(_, _)).Times(0);

  MockScheduler MS;
  auto Lock = MS.acquireGraphReadLock();
  detail::EnqueueResultT Res;
  bool Enqueued = MockScheduler::enqueueCommand(&A, Res, detail::NON_BLOCKING);
  ASSERT_FALSE(Enqueued) << "Blocked command should not be enqueued\n";
  ASSERT_EQ(detail::EnqueueResultT::SyclEnqueueBlocked, Res.MResult)
      << "Result of enqueueing blocked command should be BLOCKED.\n";
  ASSERT_EQ(&C, Res.MCmd) << "Expected different failed command.\n";
}

TEST_F(SchedulerTest, EnqueueBlockedCommandEarlyExit) {
  MockCommand A(detail::getSyclObjImpl(MQueue));
  A.MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueBlocked;
  A.MIsBlockable = true;

  MockCommand B(detail::getSyclObjImpl(MQueue));
  B.MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueReady;
  B.MRetVal = CL_OUT_OF_RESOURCES;

  addEdge(&A, &B, nullptr);

  // We have such a graph:
  //
  //   A -> B
  //
  // If A is blocked, we should not try to enqueue B.

  EXPECT_CALL(A, enqueue(_, _)).Times(0);
  EXPECT_CALL(B, enqueue(_, _)).Times(0);

  MockScheduler MS;
  auto Lock = MS.acquireGraphReadLock();
  detail::EnqueueResultT Res;
  bool Enqueued = MockScheduler::enqueueCommand(&A, Res, detail::NON_BLOCKING);
  ASSERT_FALSE(Enqueued) << "Blocked command should not be enqueued\n";
  ASSERT_EQ(detail::EnqueueResultT::SyclEnqueueBlocked, Res.MResult)
      << "Result of enqueueing blocked command should be BLOCKED.\n";
  ASSERT_EQ(&A, Res.MCmd) << "Expected different failed command.\n";

  // But if the enqueue type is blocking we should not exit early.

  EXPECT_CALL(A, enqueue(_, _)).Times(0);
  EXPECT_CALL(B, enqueue(_, _)).Times(1);

  Enqueued = MockScheduler::enqueueCommand(&A, Res, detail::BLOCKING);
  ASSERT_FALSE(Enqueued) << "Blocked command should not be enqueued\n";
  ASSERT_EQ(detail::EnqueueResultT::SyclEnqueueFailed, Res.MResult)
      << "Result of enqueueing blocked command should be BLOCKED.\n";
  ASSERT_EQ(&B, Res.MCmd) << "Expected different failed command.\n";
}

// This unit test is for workaround described in GraphProcessor::enqueueCommand
// method.
TEST_F(SchedulerTest, EnqueueHostDependency) {
  MockCommand A(detail::getSyclObjImpl(MQueue));
  A.MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueReady;
  A.MIsBlockable = true;
  A.MRetVal = CL_SUCCESS;

  MockCommand B(detail::getSyclObjImpl(MQueue));
  B.MEnqueueStatus = detail::EnqueueResultT::SyclEnqueueReady;
  B.MIsBlockable = true;
  B.MRetVal = CL_SUCCESS;

  cl::sycl::detail::EventImplPtr DepEvent{
      new cl::sycl::detail::event_impl(detail::getSyclObjImpl(MQueue))};
  DepEvent->setCommand(&B);

  (void)A.addDep(DepEvent);

  // We have such a "graph":
  //
  //     A
  //     |
  //     B
  //
  // A depends on B. B is host command.
  // "Graph" is quoted as we don't have this dependency in MDeps. Instead, we
  // have this dependecy as result of handler::depends_on() call.

  EXPECT_CALL(A, enqueue(_, _)).Times(1);
  EXPECT_CALL(B, enqueue(_, _)).Times(1);

  MockScheduler MS;
  auto Lock = MS.acquireGraphReadLock();
  detail::EnqueueResultT Res;
  bool Enqueued = MockScheduler::enqueueCommand(&A, Res, detail::NON_BLOCKING);
  ASSERT_TRUE(Enqueued) << "The command should be enqueued\n";
  ASSERT_EQ(detail::EnqueueResultT::SyclEnqueueSuccess, Res.MResult)
      << "Enqueue operation should return successfully.\n";
}
