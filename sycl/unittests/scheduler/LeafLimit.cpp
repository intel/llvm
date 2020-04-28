//==---------------- LeafLimit.cpp --- Scheduler unit tests ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

#include <CL/sycl.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

using namespace cl::sycl;

TEST_F(SchedulerTest, LeafLimit) {
  TestScheduler TS;

  buffer<int, 1> Buf(range<1>(1));
  detail::Requirement FakeReq{{0, 0, 0},
                              {0, 0, 0},
                              {0, 0, 0},
                              access::mode::read_write,
                              detail::getSyclObjImpl(Buf).get(),
                              0,
                              0,
                              0};
  FakeCommand *FakeDepCmd =
      new FakeCommand(detail::getSyclObjImpl(MQueue), FakeReq);
  detail::MemObjRecord *Rec =
      TS.getOrInsertMemObjRecord(detail::getSyclObjImpl(MQueue), &FakeReq);

  // Create commands that will be added as leaves exceeding the limit by 1
  std::vector<FakeCommand *> LeavesToAdd;
  for (std::size_t i = 0; i < Rec->MWriteLeaves.capacity() + 1; ++i) {
    LeavesToAdd.push_back(
        new FakeCommand(detail::getSyclObjImpl(MQueue), FakeReq));
  }
  // Create edges: all soon-to-be leaves are direct users of FakeDep
  for (auto Leaf : LeavesToAdd) {
    FakeDepCmd->addUser(Leaf);
    Leaf->addDep(detail::DepDesc{FakeDepCmd, Leaf->getRequirement(), nullptr});
  }
  // Add edges as leaves and exceed the leaf limit
  for (auto LeafPtr : LeavesToAdd) {
    TS.addNodeToLeaves(Rec, LeafPtr);
  }
  // Check that the oldest leaf has been removed from the leaf list
  // and added as a dependency of the newest one instead
  const detail::CircularBuffer<detail::Command *> &Leaves = Rec->MWriteLeaves;
  ASSERT_TRUE(std::find(Leaves.begin(), Leaves.end(), LeavesToAdd.front()) ==
              Leaves.end());
  for (std::size_t i = 1; i < LeavesToAdd.size(); ++i) {
    assert(std::find(Leaves.begin(), Leaves.end(), LeavesToAdd[i]) !=
           Leaves.end());
  }
  FakeCommand *OldestLeaf = LeavesToAdd.front();
  FakeCommand *NewestLeaf = LeavesToAdd.back();
  ASSERT_EQ(OldestLeaf->MUsers.size(), 1U);
  EXPECT_GT(OldestLeaf->MUsers.count(NewestLeaf), 0U);
  ASSERT_EQ(NewestLeaf->MDeps.size(), 2U);
  EXPECT_TRUE(std::any_of(
      NewestLeaf->MDeps.begin(), NewestLeaf->MDeps.end(),
      [&](const detail::DepDesc &DD) { return DD.MDepCommand == OldestLeaf; }));

  FakeDepCmd->getEvent()->setComplete();
  for (FakeCommand *Cmd : LeavesToAdd)
    Cmd->getEvent()->setComplete();
}
