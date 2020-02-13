// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out
#include <CL/sycl.hpp>

#include <algorithm>
#include <vector>

#include "SchedulerTestUtils.hpp"

using namespace cl::sycl;

// This test checks regular execution graph cleanup at host-device
// synchronization points
int main() {
  TestScheduler TS;
  queue Queue;
  buffer<int, 1> BufA(range<1>(1));
  buffer<int, 1> BufB(range<1>(1));
  buffer<int, 1> BufC(range<1>(1));
  detail::Requirement FakeReqA = getFakeRequirement(BufA);
  detail::Requirement FakeReqB = getFakeRequirement(BufB);
  detail::Requirement FakeReqC = getFakeRequirement(BufC);
  detail::MemObjRecord *RecC =
      TS.getOrInsertMemObjRecord(detail::getSyclObjImpl(Queue), &FakeReqC);

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
  detail::AllocaCommand AllocaA{detail::getSyclObjImpl(Queue), FakeReqA};
  detail::AllocaCommand AllocaB{detail::getSyclObjImpl(Queue), FakeReqB};

  int NInnerCommandsAlive = 3;
  std::function<void()> Callback = [&]() { --NInnerCommandsAlive; };

  FakeCommand *InnerC = new FakeCommandWithCallback(
      detail::getSyclObjImpl(Queue), FakeReqA, Callback);
  addEdge(InnerC, &AllocaA, &AllocaA);

  FakeCommand LeafB{detail::getSyclObjImpl(Queue), FakeReqB};
  addEdge(&LeafB, &AllocaB, &AllocaB);
  TS.addNodeToLeaves(RecC, &LeafB);

  FakeCommand LeafA{detail::getSyclObjImpl(Queue), FakeReqA};
  addEdge(&LeafA, InnerC, &AllocaA);
  TS.addNodeToLeaves(RecC, &LeafA);

  FakeCommand *InnerB = new FakeCommandWithCallback(
      detail::getSyclObjImpl(Queue), FakeReqB, Callback);
  addEdge(InnerB, &LeafB, &AllocaB);

  FakeCommand *InnerA = new FakeCommandWithCallback(
      detail::getSyclObjImpl(Queue), FakeReqA, Callback);
  addEdge(InnerA, &LeafA, &AllocaA);
  addEdge(InnerA, InnerB, &AllocaB);

  TS.cleanupFinishedCommands(InnerA);
  TS.removeRecordForMemObj(detail::getSyclObjImpl(BufC).get());

  assert(NInnerCommandsAlive == 0);

  assert(LeafA.MDeps.size() == 1);
  assert(LeafA.MDeps[0].MDepCommand == &AllocaA);
  assert(AllocaA.MUsers.size() == 1);
  assert(*AllocaA.MUsers.begin() == &LeafA);

  assert(LeafB.MDeps.size() == 1);
  assert(LeafB.MDeps[0].MDepCommand == &AllocaB);
  assert(AllocaB.MUsers.size() == 1);
  assert(*AllocaB.MUsers.begin() == &LeafB);
}
