//==-------- CircularBufferExtended.cpp --- Scheduler unit tests -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTestUtils.hpp"

#include <CL/sycl.hpp>
#include <detail/scheduler/circular_buffer_extended.hpp>
#include <gtest/gtest.h>
#include <memory>

using namespace cl::sycl::detail;

class CircularBufferExtendedTest : public ::testing::Test {
};

std::shared_ptr<Command> createGenericCommand(
    const std::shared_ptr<queue_impl> &Q) {
  return std::shared_ptr<Command>{new MockCommand(Q, Command::RUN_CG)};
}

std::shared_ptr<Command> createEmptyCommand(
    const std::shared_ptr<queue_impl> &Q) {
  return std::shared_ptr<Command>{new MockCommand(Q, Command::EMPTY_TASK)};
}

TEST_F(CircularBufferExtendedTest, PushBack) {
  using GenericCommandsT = CircularBufferExtended::GenericCommandsT;
  //using HostAccessorCommandsT = CircularBufferExtended::HostAccessorCommandsT;

  static constexpr size_t GenericCmdsCapacity = 8;

  size_t TimesGenericWasFull;

  CircularBufferExtended::IfGenericIsFullF IfGenericIsFull =
      [&](Command *, MemObjRecord *, GenericCommandsT &) {
        ++TimesGenericWasFull;
      };
  CircularBufferExtended::AllocateDependencyF AllocateDependency =
      [](Command *, Command *, MemObjRecord *) {
      };

  // add only generic commands
  {
    sycl::device HostDevice;
    std::shared_ptr<queue_impl> Q(new queue_impl(
        getSyclObjImpl(HostDevice), /*AsyncHandler=*/{},
        /*PropList=*/{}));

    CircularBufferExtended CBE = CircularBufferExtended(GenericCmdsCapacity,
        IfGenericIsFull, AllocateDependency);
    std::vector<std::shared_ptr<Command>> Cmds;

    TimesGenericWasFull = 0;

    for (size_t Idx = 0; Idx < GenericCmdsCapacity * 2; ++Idx) {
      Cmds.push_back(createGenericCommand(Q));

      CBE.push_back(Cmds.back().get(), nullptr);
    }

    ASSERT_EQ(TimesGenericWasFull, GenericCmdsCapacity)
        << "IfGenericIsFull call count mismatch.";

    ASSERT_EQ(CBE.getGenericCommands().size(), GenericCmdsCapacity)
        << "Generic commands container size overflow";

    ASSERT_EQ(CBE.getHostAccessorCommands().size(), 0ul)
        << "Host accessor commands container isn't emptym but it should be.";
  }
}

