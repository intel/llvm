//==----------- FinishedCmdCleanup.cpp --- Scheduler unit tests ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

#include <CL/sycl.hpp>
#include <detail/event_impl.hpp>
#include <detail/scheduler/scheduler.hpp>

#include <gtest/gtest.h>

using namespace cl::sycl;

TEST_F(SchedulerTest, FinishedCmdCleanup) {
  TestScheduler TS;
  buffer<int, 1> BufA(range<1>(1));
  buffer<int, 1> BufB(range<1>(1));
  buffer<int, 1> BufC(range<1>(1));
  detail::Requirement FakeReqA = getFakeRequirement(BufA);
  detail::Requirement FakeReqB = getFakeRequirement(BufB);
  detail::Requirement FakeReqC = getFakeRequirement(BufC);
  detail::MemObjRecord *RecC =
      TS.getOrInsertMemObjRecord(detail::getSyclObjImpl(MQueue), &FakeReqC);

  // Create a graph and check that all inner nodes have been deleted and
  // their users have had the corresponding dependency replaced with a
  // dependency on the alloca. The graph should undergo the following
  // transformation:
  // +---------+     +---------+              +---------++---------+
  // |  LeafA  | <-- | InnerA  |              |  LeafA  ||  LeafB  |
  // +---------+     +---------+              +---------++---------+
  //   |               |                        |          |
  //   |               |             ===>       |          |
  //   v               v                        v          v
  // +---------+     +---------+              +---------++---------+
  // | InnerC  |     | InnerB  |              | AllocaA || AllocaB |
  // +---------+     +---------+              +---------++---------+
  //   |               |
  //   |               |
  //   v               v
  // +---------+     +---------+
  // | AllocaA |     |  LeafB  |
  // +---------+     +---------+
  //                   |
  //                   |
  //                   v
  //                 +---------+
  //                 | AllocaB |
  //                 +---------+
  detail::AllocaCommand AllocaA{detail::getSyclObjImpl(MQueue), FakeReqA};
  detail::AllocaCommand AllocaB{detail::getSyclObjImpl(MQueue), FakeReqB};

  int NInnerCommandsAlive = 3;
  std::function<void()> Callback = [&]() { --NInnerCommandsAlive; };

  FakeCommand *InnerC = new FakeCommandWithCallback(
      detail::getSyclObjImpl(MQueue), FakeReqA, Callback);
  addEdge(InnerC, &AllocaA, &AllocaA);

  FakeCommand LeafB{detail::getSyclObjImpl(MQueue), FakeReqB};
  addEdge(&LeafB, &AllocaB, &AllocaB);
  TS.addNodeToLeaves(RecC, &LeafB);

  FakeCommand LeafA{detail::getSyclObjImpl(MQueue), FakeReqA};
  addEdge(&LeafA, InnerC, &AllocaA);
  TS.addNodeToLeaves(RecC, &LeafA);

  FakeCommand *InnerB = new FakeCommandWithCallback(
      detail::getSyclObjImpl(MQueue), FakeReqB, Callback);
  addEdge(InnerB, &LeafB, &AllocaB);

  FakeCommand *InnerA = new FakeCommandWithCallback(
      detail::getSyclObjImpl(MQueue), FakeReqA, Callback);
  addEdge(InnerA, &LeafA, &AllocaA);
  addEdge(InnerA, InnerB, &AllocaB);

  std::shared_ptr<detail::event_impl> Event{new detail::event_impl{}};
  Event->setCommand(InnerA);
  TS.cleanupFinishedCommands(Event);
  TS.removeRecordForMemObj(detail::getSyclObjImpl(BufC).get());

  EXPECT_EQ(NInnerCommandsAlive, 0);

  ASSERT_EQ(LeafA.MDeps.size(), 1U);
  EXPECT_EQ(LeafA.MDeps[0].MDepCommand, &AllocaA);
  ASSERT_EQ(AllocaA.MUsers.size(), 1U);
  EXPECT_EQ(*AllocaA.MUsers.begin(), &LeafA);

  ASSERT_EQ(LeafB.MDeps.size(), 1U);
  EXPECT_EQ(LeafB.MDeps[0].MDepCommand, &AllocaB);
  ASSERT_EQ(AllocaB.MUsers.size(), 1U);
  EXPECT_EQ(*AllocaB.MUsers.begin(), &LeafB);
}
