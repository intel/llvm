//==---------------- LeafLimit.cpp --- Scheduler unit tests ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

#include <detail/config.hpp>
#include <helpers/ScopedEnvVar.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

using namespace cl::sycl;

inline constexpr auto DisablePostEnqueueCleanupName =
    "SYCL_DISABLE_POST_ENQUEUE_CLEANUP";

// Checks that scheduler's (or graph-builder's) addNodeToLeaves method works
// correctly with dependency tracking when leaf-limit for generic commands is
// overflowed.
TEST_F(SchedulerTest, LeafLimit) {
  // All of the mock commands are owned on the test side, prevent post enqueue
  // cleanup from deleting some of them.
  unittest::ScopedEnvVar DisabledCleanup{
      DisablePostEnqueueCleanupName, "1",
      detail::SYCLConfig<detail::SYCL_DISABLE_POST_ENQUEUE_CLEANUP>::reset};
  cl::sycl::queue HQueue(host_selector{});
  MockScheduler MS;
  std::vector<std::unique_ptr<MockCommand>> LeavesToAdd;
  std::unique_ptr<MockCommand> MockDepCmd;

  buffer<int, 1> Buf(range<1>(1));
  detail::Requirement MockReq = getMockRequirement(Buf);

  MockDepCmd =
      std::make_unique<MockCommand>(detail::getSyclObjImpl(MQueue), MockReq);
  std::vector<detail::Command *> AuxCmds;
  detail::MemObjRecord *Rec = MS.getOrInsertMemObjRecord(
      detail::getSyclObjImpl(MQueue), &MockReq, AuxCmds);

  // Create commands that will be added as leaves exceeding the limit by 1
  for (std::size_t i = 0; i < Rec->MWriteLeaves.genericCommandsCapacity() + 1;
       ++i) {
    LeavesToAdd.push_back(
        std::make_unique<MockCommand>(detail::getSyclObjImpl(MQueue), MockReq));
  }
  // Create edges: all soon-to-be leaves are direct users of MockDep
  std::vector<detail::Command *> ToCleanUp;
  for (auto &Leaf : LeavesToAdd) {
    MockDepCmd->addUser(Leaf.get());
    (void)Leaf->addDep(
        detail::DepDesc{MockDepCmd.get(), Leaf->getRequirement(), nullptr},
        ToCleanUp);
  }
  std::vector<cl::sycl::detail::Command *> ToEnqueue;
  // Add edges as leaves and exceed the leaf limit
  for (auto &LeafPtr : LeavesToAdd) {
    MS.addNodeToLeaves(Rec, LeafPtr.get(), access::mode::write, ToEnqueue);
  }
  // Check that the oldest leaf has been removed from the leaf list
  // and added as a dependency of the newest one instead
  const detail::CircularBuffer<detail::Command *> &Leaves =
      Rec->MWriteLeaves.getGenericCommands();
  ASSERT_TRUE(std::find(Leaves.begin(), Leaves.end(),
                        LeavesToAdd.front().get()) == Leaves.end());
  for (std::size_t i = 1; i < LeavesToAdd.size(); ++i) {
    assert(std::find(Leaves.begin(), Leaves.end(), LeavesToAdd[i].get()) !=
           Leaves.end());
  }
  MockCommand *OldestLeaf = LeavesToAdd.front().get();
  MockCommand *NewestLeaf = LeavesToAdd.back().get();
  ASSERT_EQ(OldestLeaf->MUsers.size(), 1U);
  EXPECT_GT(OldestLeaf->MUsers.count(NewestLeaf), 0U);
  ASSERT_EQ(NewestLeaf->MDeps.size(), 2U);
  EXPECT_TRUE(std::any_of(
      NewestLeaf->MDeps.begin(), NewestLeaf->MDeps.end(),
      [&](const detail::DepDesc &DD) { return DD.MDepCommand == OldestLeaf; }));
}
