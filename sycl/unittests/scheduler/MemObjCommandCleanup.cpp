//==---------- MemObjCommandCleanup.cpp --- Scheduler unit tests -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

using namespace cl::sycl;

TEST_F(SchedulerTest, MemObjCommandCleanupAllocaUsers) {
  MockScheduler MS;
  buffer<int, 1> BufA(range<1>(1));
  buffer<int, 1> BufB(range<1>(1));
  detail::Requirement MockReqA = getMockRequirement(BufA);
  detail::Requirement MockReqB = getMockRequirement(BufB);
  std::vector<detail::Command *> AuxCmds;
  detail::MemObjRecord *RecA = MS.getOrInsertMemObjRecord(
      detail::getSyclObjImpl(MQueue), &MockReqA, AuxCmds);

  // Create 2 fake allocas, one of which will be cleaned up
  detail::AllocaCommand *MockAllocaA =
      new detail::AllocaCommand(detail::getSyclObjImpl(MQueue), MockReqA);
  std::unique_ptr<detail::AllocaCommand> MockAllocaB{
      new detail::AllocaCommand(detail::getSyclObjImpl(MQueue), MockReqB)};
  RecA->MAllocaCommands.push_back(MockAllocaA);

  // Create a direct user of both allocas
  std::unique_ptr<MockCommand> MockDirectUser{
      new MockCommand(detail::getSyclObjImpl(MQueue), MockReqA)};
  addEdge(MockDirectUser.get(), MockAllocaA, MockAllocaA);
  addEdge(MockDirectUser.get(), MockAllocaB.get(), MockAllocaB.get());

  // Create an indirect user of the soon-to-be deleted alloca
  bool IndirectUserDeleted = false;
  std::function<void()> Callback = [&]() { IndirectUserDeleted = true; };
  MockCommand *MockIndirectUser = new MockCommandWithCallback(
      detail::getSyclObjImpl(MQueue), MockReqA, Callback);
  addEdge(MockIndirectUser, MockDirectUser.get(), MockAllocaA);

  MS.cleanupCommandsForRecord(RecA);
  MS.removeRecordForMemObj(detail::getSyclObjImpl(BufA).get());

  // Check that the direct user has been left with the second alloca
  // as the only dependency, while the indirect user has been cleaned up.
  ASSERT_EQ(MockDirectUser->MUsers.size(), 0U);
  ASSERT_EQ(MockDirectUser->MDeps.size(), 1U);
  EXPECT_EQ(MockDirectUser->MDeps[0].MDepCommand, MockAllocaB.get());
  EXPECT_TRUE(IndirectUserDeleted);
}

TEST_F(SchedulerTest, MemObjCommandCleanupAllocaDeps) {
  MockScheduler MS;
  buffer<int, 1> Buf(range<1>(1));
  detail::Requirement MockReq = getMockRequirement(Buf);
  std::vector<detail::Command *> AuxCmds;
  detail::MemObjRecord *MemObjRec = MS.getOrInsertMemObjRecord(
      detail::getSyclObjImpl(MQueue), &MockReq, AuxCmds);

  // Create a fake alloca.
  detail::AllocaCommand *MockAllocaCmd =
      new detail::AllocaCommand(detail::getSyclObjImpl(MQueue), MockReq);
  MemObjRec->MAllocaCommands.push_back(MockAllocaCmd);

  // Add another mock command and add MockAllocaCmd as its user.
  MockCommand DepCmd(detail::getSyclObjImpl(MQueue), MockReq);
  addEdge(MockAllocaCmd, &DepCmd, nullptr);

  // Check that DepCmd.MUsers size reflect the dependency properly.
  ASSERT_EQ(DepCmd.MUsers.size(), 1U);
  ASSERT_EQ(DepCmd.MUsers.count(MockAllocaCmd), 1U);

  MS.cleanupCommandsForRecord(MemObjRec);
  MS.removeRecordForMemObj(detail::getSyclObjImpl(Buf).get());

  // Check that DepCmd has its MUsers field cleared.
  ASSERT_EQ(DepCmd.MUsers.size(), 0U);
}
