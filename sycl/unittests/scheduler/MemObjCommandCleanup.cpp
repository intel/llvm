//==---------- MemObjCommandCleanup.cpp --- Scheduler unit tests -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTest.hpp"
#include "SchedulerTestUtils.hpp"

#include <CL/sycl.hpp>
#include <detail/scheduler/scheduler.hpp>

#include <gtest/gtest.h>

using namespace cl::sycl;

TEST_F(SchedulerTest, MemObjCommandCleanup) {
  TestScheduler TS;
  buffer<int, 1> BufA(range<1>(1));
  buffer<int, 1> BufB(range<1>(1));
  detail::Requirement FakeReqA = getFakeRequirement(BufA);
  detail::Requirement FakeReqB = getFakeRequirement(BufB);
  detail::MemObjRecord *RecA =
      TS.getOrInsertMemObjRecord(detail::getSyclObjImpl(MQueue), &FakeReqA);

  // Create 2 fake allocas, one of which will be cleaned up
  detail::AllocaCommand *FakeAllocaA =
      new detail::AllocaCommand(detail::getSyclObjImpl(MQueue), FakeReqA);
  std::unique_ptr<detail::AllocaCommand> FakeAllocaB{
      new detail::AllocaCommand(detail::getSyclObjImpl(MQueue), FakeReqB)};
  RecA->MAllocaCommands.push_back(FakeAllocaA);

  // Create a direct user of both allocas
  std::unique_ptr<FakeCommand> FakeDirectUser{
      new FakeCommand(detail::getSyclObjImpl(MQueue), FakeReqA)};
  addEdge(FakeDirectUser.get(), FakeAllocaA, FakeAllocaA);
  addEdge(FakeDirectUser.get(), FakeAllocaB.get(), FakeAllocaB.get());

  // Create an indirect user of the soon-to-be deleted alloca
  bool IndirectUserDeleted = false;
  std::function<void()> Callback = [&]() { IndirectUserDeleted = true; };
  FakeCommand *FakeIndirectUser = new FakeCommandWithCallback(
      detail::getSyclObjImpl(MQueue), FakeReqA, Callback);
  addEdge(FakeIndirectUser, FakeDirectUser.get(), FakeAllocaA);

  TS.cleanupCommandsForRecord(RecA);
  TS.removeRecordForMemObj(detail::getSyclObjImpl(BufA).get());

  // Check that the direct user has been left with the second alloca
  // as the only dependency, while the indirect user has been cleaned up.
  ASSERT_EQ(FakeDirectUser->MUsers.size(), 0U);
  ASSERT_EQ(FakeDirectUser->MDeps.size(), 1U);
  EXPECT_EQ(FakeDirectUser->MDeps[0].MDepCommand, FakeAllocaB.get());
  EXPECT_TRUE(IndirectUserDeleted);
}
