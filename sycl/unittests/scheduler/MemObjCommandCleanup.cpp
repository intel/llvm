//==---------- MemObjCommandCleanup.cpp --- Scheduler unit tests -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

#include <helpers/UrMock.hpp>

#include <detail/buffer_impl.hpp>

using namespace sycl;

TEST_F(SchedulerTest, MemObjCommandCleanupAllocaUsers) {
  sycl::unittest::UrMock<> Mock;
  sycl::queue Q{sycl::platform().get_devices()[0], MAsyncHandler};
  sycl::detail::queue_impl &QueueImpl = *detail::getSyclObjImpl(Q);

  MockScheduler MS;
  buffer<int, 1> BufA(range<1>(1));
  buffer<int, 1> BufB(range<1>(1));
  detail::Requirement MockReqA = getMockRequirement(BufA);
  detail::Requirement MockReqB = getMockRequirement(BufB);
  detail::MemObjRecord *RecA =
      MS.getOrInsertMemObjRecord(&QueueImpl, &MockReqA);

  // Create 2 fake allocas, one of which will be cleaned up
  detail::AllocaCommand *MockAllocaA =
      new detail::AllocaCommand(&QueueImpl, MockReqA);
  std::unique_ptr<detail::AllocaCommand> MockAllocaB{
      new detail::AllocaCommand(&QueueImpl, MockReqB)};
  RecA->MAllocaCommands.push_back(MockAllocaA);

  // Create a direct user of both allocas
  std::unique_ptr<MockCommand> MockDirectUser{
      new MockCommand(&QueueImpl, MockReqA)};
  addEdge(MockDirectUser.get(), MockAllocaA, MockAllocaA);
  addEdge(MockDirectUser.get(), MockAllocaB.get(), MockAllocaB.get());

  // Create an indirect user of the soon-to-be deleted alloca
  bool IndirectUserDeleted = false;
  std::function<void()> Callback = [&]() { IndirectUserDeleted = true; };
  MockCommand *MockIndirectUser =
      new MockCommandWithCallback(&QueueImpl, MockReqA, Callback);
  addEdge(MockIndirectUser, MockDirectUser.get(), MockAllocaA);

  MS.cleanupCommandsForRecord(RecA);
  MS.removeRecordForMemObj(&*detail::getSyclObjImpl(BufA));

  // Check that the direct user has been left with the second alloca
  // as the only dependency, while the indirect user has been cleaned up.
  ASSERT_EQ(MockDirectUser->MUsers.size(), 0U);
  ASSERT_EQ(MockDirectUser->MDeps.size(), 1U);
  EXPECT_EQ(MockDirectUser->MDeps[0].MDepCommand, MockAllocaB.get());
  EXPECT_TRUE(IndirectUserDeleted);
}

TEST_F(SchedulerTest, MemObjCommandCleanupAllocaDeps) {
  sycl::unittest::UrMock<> Mock;
  sycl::queue Q{sycl::platform().get_devices()[0], MAsyncHandler};
  sycl::detail::queue_impl &QueueImpl = *detail::getSyclObjImpl(Q);

  MockScheduler MS;
  buffer<int, 1> Buf(range<1>(1));
  detail::Requirement MockReq = getMockRequirement(Buf);
  detail::MemObjRecord *MemObjRec =
      MS.getOrInsertMemObjRecord(&QueueImpl, &MockReq);

  // Create a fake alloca.
  detail::AllocaCommand *MockAllocaCmd =
      new detail::AllocaCommand(&QueueImpl, MockReq);
  MemObjRec->MAllocaCommands.push_back(MockAllocaCmd);

  // Add another mock command and add MockAllocaCmd as its user.
  MockCommand DepCmd(&QueueImpl, MockReq);
  addEdge(MockAllocaCmd, &DepCmd, nullptr);

  // Check that DepCmd.MUsers size reflect the dependency properly.
  ASSERT_EQ(DepCmd.MUsers.size(), 1U);
  ASSERT_EQ(DepCmd.MUsers.count(MockAllocaCmd), 1U);

  MS.cleanupCommandsForRecord(MemObjRec);
  MS.removeRecordForMemObj(&*detail::getSyclObjImpl(Buf));

  // Check that DepCmd has its MUsers field cleared.
  ASSERT_EQ(DepCmd.MUsers.size(), 0U);
}
