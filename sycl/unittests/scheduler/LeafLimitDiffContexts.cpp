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
#include <helpers/PiMock.hpp>
#include <helpers/ScopedEnvVar.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

using namespace sycl;

inline constexpr auto DisableCleanupName =
    "SYCL_DISABLE_EXECUTION_GRAPH_CLEANUP";

// Checks that scheduler's (or graph-builder's) addNodeToLeaves method works
// correctly with dependency tracking when leaf-limit for generic commands is
// overflowed.
// Checks that in case of different contexts for deleted leaf and a new one
// ConnectCmd will be created and scheduler will build the following dependency
// structure: NewLeaf->ConnectCmd->OldLeaf
TEST_F(SchedulerTest, LeafLimitDiffContexts) {
  // All of the mock commands are owned on the test side, prevent post enqueue
  // cleanup from deleting some of them.
  unittest::ScopedEnvVar DisabledCleanup{
      DisableCleanupName, "1",
      detail::SYCLConfig<detail::SYCL_DISABLE_EXECUTION_GRAPH_CLEANUP>::reset};

  // Ensure the mock plugin has been initialized prior to selecting a device.
  unittest::PiMock::EnsureMockPluginInitialized();

  device Device;
  struct QueueRelatedObjects {
    context Context;
    queue Queue;
    std::unique_ptr<MockCommand> DepCmd;
    detail::MemObjRecord *Rec;
    detail::AllocaCommandBase *AllocaCmd;

    QueueRelatedObjects(const device &Dev)
        : Context(Dev), Queue(Context, Dev), DepCmd(), Rec(nullptr),
          AllocaCmd(nullptr) {}

    void InitializeUtils(detail::Requirement &MockReq, MockScheduler &MS) {

      Rec = MS.getOrInsertMemObjRecord(detail::getSyclObjImpl(Queue), &MockReq);
      // Creating Alloca on both - device and host contexts (will be created in
      // real case in insertMemMove for example) It is done to avoid extra
      // AllocCmd insertion during ConnectCmd insertion
      std::vector<detail::Command *> ToEnqueue;
      AllocaCmd = MS.getOrCreateAllocaForReq(
          Rec, &MockReq, detail::getSyclObjImpl(Queue), ToEnqueue);
      std::ignore =
          MS.getOrCreateAllocaForReq(Rec, &MockReq, nullptr, ToEnqueue);
      DepCmd =
          std::make_unique<MockCommand>(detail::getSyclObjImpl(Queue), MockReq);
    }
  };

  // Creating 2 queues with different context objects
  QueueRelatedObjects ExtQueue1(Device), ExtQueue2(Device);
  MockScheduler MS;
  std::vector<std::unique_ptr<MockCommand>> AddedLeaves;

  buffer<int, 1> Buf(range<1>(1));
  detail::Requirement MockReq = getMockRequirement(Buf);

  ExtQueue1.InitializeUtils(MockReq, MS);
  ExtQueue2.InitializeUtils(MockReq, MS);

  size_t CommandsCapacity =
      ExtQueue1.Rec->MWriteLeaves.genericCommandsCapacity();

  // Adds leaf with 1 deps to buffer
  auto AddLeafWithDeps = [&AddedLeaves, &MockReq,
                          &MS](const QueueRelatedObjects &QueueStuff) {
    auto NewLeaf = std::make_unique<MockCommand>(
        detail::getSyclObjImpl(QueueStuff.Queue), MockReq);
    // Create edges: all soon-to-be leaves are direct users of MockDep
    std::vector<detail::Command *> ToCleanUp;
    (void)NewLeaf->addDep(detail::DepDesc{QueueStuff.DepCmd.get(), &MockReq,
                                          QueueStuff.AllocaCmd},
                          ToCleanUp);
    QueueStuff.DepCmd->addUser(NewLeaf.get());

    std::vector<detail::Command *> ToEnqueue;
    MS.addNodeToLeaves(QueueStuff.Rec, NewLeaf.get(), access::mode::write,
                       ToEnqueue);
    AddedLeaves.push_back(std::move(NewLeaf));
  };

  // Create commands that will be added as leaves up to the limit
  for (std::size_t i = 0; i < CommandsCapacity; ++i) {
    AddLeafWithDeps(ExtQueue1);
  }
  // Adding extra command on different to exceed buffer limit
  // The command #0 and command #8 must be on different queues to insert connect
  // command
  AddLeafWithDeps(ExtQueue2);

  // Check that the oldest leaf #0 has been removed from the leaf list
  const detail::CircularBuffer<detail::Command *> &Leaves =
      ExtQueue1.Rec->MWriteLeaves.getGenericCommands();
  ASSERT_TRUE(std::find(Leaves.begin(), Leaves.end(),
                        AddedLeaves.front().get()) == Leaves.end());
  // Check that another leaves #1...#7 that should not be removed are present in
  // buffer
  for (std::size_t i = 1; i < AddedLeaves.size(); ++i) {
    assert(std::find(Leaves.begin(), Leaves.end(), AddedLeaves[i].get()) !=
           Leaves.end());
  }

  // Check NewLeaf->ConnectCmd->OldLeaf structure
  MockCommand *OldestLeaf = AddedLeaves.front().get();
  MockCommand *NewestLeaf = AddedLeaves.back().get();
  // The only user for oldLeaf must be ConnectCmd
  ASSERT_EQ(OldestLeaf->MUsers.size(), 1U);
  // No direct connection between OldLeaf and newLeaf, only via ConnectCmd
  EXPECT_EQ(OldestLeaf->MUsers.count(NewestLeaf), 0U);
  // 2 deps for NewLeaf: 1 dep command and connect cmd - no OldLeaf direct
  // dependency
  ASSERT_EQ(NewestLeaf->MDeps.size(), 2U);
  EXPECT_FALSE(std::any_of(
      NewestLeaf->MDeps.begin(), NewestLeaf->MDeps.end(),
      [&](const detail::DepDesc &DD) { return DD.MDepCommand == OldestLeaf; }));
  // Check NewLeaf dependencies in depth by MUsers
  auto ConnectCmdIt = OldestLeaf->MUsers.begin();
  ASSERT_EQ((*ConnectCmdIt)->MUsers.size(), 1U);
  EXPECT_TRUE(std::any_of(NewestLeaf->MDeps.begin(), NewestLeaf->MDeps.end(),
                          [&](const detail::DepDesc &DD) {
                            return DD.MDepCommand == (*ConnectCmdIt);
                          }));
  // ConnectCmd is created internally in scheduler and not a mock object
  // This fact leads to active scheduler shutdown process that deletes a
  // part of commands for record we store in AddedLeaves vector.
  // We abort this process by removing record to avoid double release or
  // or memory leaks
  MS.removeRecordForMemObj(MockReq.MSYCLMemObj);
}
