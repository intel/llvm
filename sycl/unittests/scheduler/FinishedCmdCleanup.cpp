//==----------- FinishedCmdCleanup.cpp --- Scheduler unit tests ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

#include <detail/event_impl.hpp>

using namespace cl::sycl;

TEST_F(SchedulerTest, FinishedCmdCleanup) {
  MockScheduler MS;
  buffer<int, 1> BufA(range<1>(1));
  buffer<int, 1> BufB(range<1>(1));
  buffer<int, 1> BufC(range<1>(1));
  detail::Requirement MockReqA = getMockRequirement(BufA);
  detail::Requirement MockReqB = getMockRequirement(BufB);
  detail::Requirement MockReqC = getMockRequirement(BufC);
  std::vector<detail::Command *> AuxCmds;
  detail::MemObjRecord *RecC = MS.getOrInsertMemObjRecord(
      detail::getSyclObjImpl(MQueue), &MockReqC, AuxCmds);

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
  detail::AllocaCommand AllocaA{detail::getSyclObjImpl(MQueue), MockReqA};
  detail::AllocaCommand AllocaB{detail::getSyclObjImpl(MQueue), MockReqB};

  int NInnerCommandsAlive = 3;
  std::function<void()> Callback = [&]() { --NInnerCommandsAlive; };

  MockCommand *InnerC = new MockCommandWithCallback(
      detail::getSyclObjImpl(MQueue), MockReqA, Callback);
  addEdge(InnerC, &AllocaA, &AllocaA);

  std::vector<detail::Command *> ToEnqueue;

  MockCommand LeafB{detail::getSyclObjImpl(MQueue), MockReqB};
  addEdge(&LeafB, &AllocaB, &AllocaB);
  MS.addNodeToLeaves(RecC, &LeafB, access::mode::read, ToEnqueue);

  MockCommand LeafA{detail::getSyclObjImpl(MQueue), MockReqA};
  addEdge(&LeafA, InnerC, &AllocaA);
  MS.addNodeToLeaves(RecC, &LeafA, access::mode::read, ToEnqueue);

  MockCommand *InnerB = new MockCommandWithCallback(
      detail::getSyclObjImpl(MQueue), MockReqB, Callback);
  addEdge(InnerB, &LeafB, &AllocaB);

  MockCommand *InnerA = new MockCommandWithCallback(
      detail::getSyclObjImpl(MQueue), MockReqA, Callback);
  addEdge(InnerA, &LeafA, &AllocaA);
  addEdge(InnerA, InnerB, &AllocaB);

  std::shared_ptr<detail::event_impl> Event{new detail::event_impl{}};
  Event->setCommand(InnerA);
  MS.cleanupFinishedCommands(Event);
  MS.removeRecordForMemObj(detail::getSyclObjImpl(BufC).get());

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
